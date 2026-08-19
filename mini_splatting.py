"""Distributed implementation of Mini-Splatting's simplification stages."""

import torch
import torch.distributed as dist

from gaussian_renderer import (
    distributed_preprocess3dgs_and_all2all_final,
    render_mini_splatting_depth,
    render_mini_splatting_importance,
)
from gaussian_renderer.loss_distribution import load_camera_from_cpu_to_all_gpu
from gaussian_renderer.workload_division import start_strategy_final
from simple_knn._C import distCUDA2
import utils.general_utils as utils


CUDA_MULTINOMIAL_MAX_CATEGORIES = 1 << 24


def _counts(local_count):
    group = utils.DEFAULT_GROUP
    count = torch.tensor([local_count], dtype=torch.long, device="cuda")
    if group.size() == 1:
        return [int(local_count)]
    gathered = torch.empty(group.size(), dtype=torch.long, device="cuda")
    dist.all_gather_into_tensor(gathered, count, group=group)
    return [int(value) for value in gathered.cpu().tolist()]


def _gather_to_rank0(local_tensor, counts):
    group = utils.DEFAULT_GROUP
    if group.size() == 1:
        return local_tensor

    tail_shape = tuple(local_tensor.shape[1:])
    empty = lambda count: torch.empty(
        (count,) + tail_shape, dtype=local_tensor.dtype, device="cuda"
    )
    send = [
        local_tensor.contiguous() if rank == 0 else empty(0)
        for rank in range(group.size())
    ]
    recv = [
        empty(counts[source]) if group.rank() == 0 else empty(0)
        for source in range(group.size())
    ]
    dist.all_to_all(recv, send, group=group)
    return torch.cat(recv, dim=0) if group.rank() == 0 else None


def _scatter_from_rank0(global_tensor, counts, local_reference):
    group = utils.DEFAULT_GROUP
    if group.size() == 1:
        return global_tensor

    tail_shape = tuple(local_reference.shape[1:])
    empty = lambda count: torch.empty(
        (count,) + tail_shape, dtype=local_reference.dtype, device="cuda"
    )
    if group.rank() == 0:
        send = [chunk.contiguous() for chunk in global_tensor.split(counts, dim=0)]
    else:
        send = [empty(0) for _ in range(group.size())]
    recv = [
        empty(counts[group.rank()]) if source == 0 else empty(0)
        for source in range(group.size())
    ]
    dist.all_to_all(recv, send, group=group)
    return recv[0]


def _route_stats_to_owners(
    render_stats,
    screenspace_pkg,
    importance,
    area_max,
    overload_pressure,
    overloaded_touches,
    metric,
):
    group = utils.DEFAULT_GROUP
    rank = group.rank()
    sizes = screenspace_pkg["gpui_to_gpuj_imgk_size"]
    local_send_ids = screenspace_pkg["local_to_gpuj_camk_send_ids"]

    stats_to_owner = []
    offset = 0
    for owner in range(group.size()):
        count = int(sizes[owner][rank][0])
        stats_to_owner.append(render_stats[offset : offset + count].contiguous())
        offset += count
    if offset != render_stats.shape[0]:
        raise RuntimeError(
            "Mini-Splatting render statistics have invalid ownership metadata"
        )

    recv = [
        torch.empty(
            (local_send_ids[renderer][0].numel(), 5),
            dtype=torch.float32,
            device="cuda",
        )
        for renderer in range(group.size())
    ]
    if group.size() == 1:
        recv[0].copy_(stats_to_owner[0])
    else:
        dist.all_to_all(recv, stats_to_owner, group=group)

    owner_stats = torch.zeros(
        (importance.shape[0], 5), dtype=torch.float32, device="cuda"
    )
    for renderer in range(group.size()):
        ids = local_send_ids[renderer][0].to(
            device="cuda", dtype=torch.long
        ).flatten()
        if recv[renderer].shape[0] != ids.numel():
            raise RuntimeError(
                "Mini-Splatting owner routing size mismatch for renderer {}: "
                "ids={} stats={}".format(
                    renderer, ids.numel(), recv[renderer].shape[0]
                )
            )
        if ids.numel() > 0:
            min_id = int(ids.min().item())
            max_id = int(ids.max().item())
            if min_id < 0 or max_id >= owner_stats.shape[0]:
                raise RuntimeError(
                    "Mini-Splatting owner routing produced invalid Gaussian IDs "
                    "for renderer {}: range=[{}, {}], local_count={}".format(
                        renderer, min_id, max_id, owner_stats.shape[0]
                    )
                )
            owner_stats.index_add_(0, ids, recv[renderer])

    view_has_max = owner_stats[:, 2] != 0
    area_max.add_(owner_stats[:, 2])
    if metric == "outdoor":
        valid = torch.logical_and(view_has_max, owner_stats[:, 1] != 0)
        importance[valid] += owner_stats[valid, 0] / owner_stats[valid, 1]
    else:
        importance.add_(owner_stats[:, 0])
    overload_pressure.add_(owner_stats[:, 3])
    overloaded_touches.add_(owner_stats[:, 4])


def _collect_importance(
    iteration,
    train_dataset,
    gaussians,
    pipe_args,
    background,
    strategy_history,
    tile_budget=0,
):
    args = utils.get_args()
    importance = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    area_max = torch.zeros_like(importance)
    overload_pressure = torch.zeros_like(importance)
    overloaded_touches = torch.zeros_like(importance)
    local_tile_occupancies = []

    for camera in train_dataset.cameras:
        strategies, _ = start_strategy_final([camera], strategy_history)
        screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            [camera],
            gaussians,
            pipe_args,
            background,
            batched_strategies=strategies,
            iteration=iteration,
            mode="test",
        )
        batched_stats, batched_n_render = render_mini_splatting_importance(
            screenspace_pkg, strategies, tile_budget=tile_budget
        )
        render_stats = batched_stats[0]
        n_render = batched_n_render[0]
        _route_stats_to_owners(
            render_stats,
            screenspace_pkg,
            importance,
            area_max,
            overload_pressure,
            overloaded_touches,
            args.mini_splatting_imp_metric,
        )
        if tile_budget > 0 and n_render.numel() > 0:
            compute_locally = strategies[0].get_compute_locally()
            if compute_locally is not None:
                compute_locally = compute_locally.flatten()
                if compute_locally.numel() != n_render.numel():
                    raise RuntimeError(
                        "Mini-Splatting tile statistics shape mismatch: "
                        "mask={} occupancy={}".format(
                            compute_locally.numel(), n_render.numel()
                        )
                    )
                local_tile_occupancies.append(
                    n_render.flatten()[compute_locally].contiguous()
                )
        del (
            batched_stats,
            batched_n_render,
            render_stats,
            n_render,
            screenspace_pkg,
        )

    importance[area_max == 0] = 0
    tile_summary = {
        "total_tiles": 0,
        "active_tiles": 0,
        "empty_tiles": 0,
        "overloaded_tiles": 0,
        "excess_instances": 0,
        "intersections": 0,
        "max_occupancy": 0,
        "mean_occupancy": 0.0,
        "mean_active_occupancy": 0.0,
        "mean_overloaded_occupancy": 0.0,
        "overloaded_ratio": 0.0,
        "p50_occupancy": 0,
        "p90_occupancy": 0,
        "p95_occupancy": 0,
        "p99_occupancy": 0,
    }
    if tile_budget > 0:
        local_occupancies = (
            torch.cat(local_tile_occupancies)
            if local_tile_occupancies
            else torch.empty(0, dtype=torch.int32, device="cuda")
        )
        occupancy_counts = _counts(local_occupancies.numel())
        global_occupancies = _gather_to_rank0(
            local_occupancies, occupancy_counts
        )

        integer_stats = torch.zeros(11, dtype=torch.long, device="cuda")
        float_stats = torch.zeros(4, dtype=torch.float64, device="cuda")
        if utils.DEFAULT_GROUP.rank() == 0:
            total_tiles = global_occupancies.numel()
            active_occupancies = global_occupancies[global_occupancies > 0]
            overloaded_occupancies = global_occupancies[
                global_occupancies > tile_budget
            ]
            active_tiles = active_occupancies.numel()
            overloaded_tiles = overloaded_occupancies.numel()

            percentile_values = torch.zeros(
                4, dtype=torch.long, device="cuda"
            )
            if active_tiles > 0:
                sorted_active = torch.sort(active_occupancies).values
                percentile_positions = torch.tensor(
                    [
                        round(0.50 * (active_tiles - 1)),
                        round(0.90 * (active_tiles - 1)),
                        round(0.95 * (active_tiles - 1)),
                        round(0.99 * (active_tiles - 1)),
                    ],
                    dtype=torch.long,
                    device="cuda",
                )
                percentile_values = sorted_active[percentile_positions].long()

            integer_stats = torch.cat(
                (
                    torch.tensor(
                        [
                            total_tiles,
                            active_tiles,
                            total_tiles - active_tiles,
                            overloaded_tiles,
                            int(
                                torch.clamp_min(
                                    global_occupancies - tile_budget, 0
                                ).sum().item()
                            ),
                            int(global_occupancies.sum().item()),
                            int(global_occupancies.max().item())
                            if total_tiles > 0
                            else 0,
                        ],
                        dtype=torch.long,
                        device="cuda",
                    ),
                    percentile_values,
                )
            )
            float_stats = torch.tensor(
                [
                    float(global_occupancies.float().mean().item())
                    if total_tiles > 0
                    else 0.0,
                    float(active_occupancies.float().mean().item())
                    if active_tiles > 0
                    else 0.0,
                    float(overloaded_occupancies.float().mean().item())
                    if overloaded_tiles > 0
                    else 0.0,
                    100.0 * overloaded_tiles / active_tiles
                    if active_tiles > 0
                    else 0.0,
                ],
                dtype=torch.float64,
                device="cuda",
            )
        if utils.DEFAULT_GROUP.size() > 1:
            dist.broadcast(integer_stats, src=0, group=utils.DEFAULT_GROUP)
            dist.broadcast(float_stats, src=0, group=utils.DEFAULT_GROUP)

        integer_values = integer_stats.cpu().tolist()
        float_values = float_stats.cpu().tolist()
        tile_summary = {
            "total_tiles": int(integer_values[0]),
            "active_tiles": int(integer_values[1]),
            "empty_tiles": int(integer_values[2]),
            "overloaded_tiles": int(integer_values[3]),
            "excess_instances": int(integer_values[4]),
            "intersections": int(integer_values[5]),
            "max_occupancy": int(integer_values[6]),
            "p50_occupancy": int(integer_values[7]),
            "p90_occupancy": int(integer_values[8]),
            "p95_occupancy": int(integer_values[9]),
            "p99_occupancy": int(integer_values[10]),
            "mean_occupancy": float(float_values[0]),
            "mean_active_occupancy": float(float_values[1]),
            "mean_overloaded_occupancy": float(float_values[2]),
            "overloaded_ratio": float(float_values[3]),
        }
    return importance, overload_pressure, overloaded_touches, tile_summary


def _weighted_sample_without_replacement(weights, sample_count, generator):
    """Sample weighted indices without CUDA multinomial's category limit."""
    if sample_count == weights.numel():
        return torch.arange(weights.numel(), device=weights.device)
    if weights.numel() <= CUDA_MULTINOMIAL_MAX_CATEGORIES:
        return torch.multinomial(
            weights,
            sample_count,
            replacement=False,
            generator=generator,
        )

    # Exponential keys induce weighted sampling without replacement while
    # avoiding CUDA multinomial's 2^24 category limit.
    keys = torch.empty_like(weights)
    keys.exponential_(generator=generator)
    keys.div_(weights)
    return torch.topk(keys, sample_count, largest=False, sorted=False).indices


def _source_sampling_mask(local_importance, sampling_factor, seed):
    counts = _counts(local_importance.numel())
    global_importance = _gather_to_rank0(local_importance, counts)
    local_mask_template = torch.empty(
        local_importance.shape[0], dtype=torch.uint8, device="cuda"
    )

    global_mask = None
    sampled_count = 0
    nonzero_count = 0
    if utils.DEFAULT_GROUP.rank() == 0:
        positive_ids = torch.nonzero(global_importance > 0, as_tuple=False).flatten()
        nonzero_count = int(positive_ids.numel())
        sampled_count = int(nonzero_count * sampling_factor)
        if sampled_count > 0:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            sampled_ids = _weighted_sample_without_replacement(
                global_importance[positive_ids],
                sampled_count,
                generator=generator,
            )
            indices = positive_ids[sampled_ids]
            global_mask = torch.zeros_like(global_importance, dtype=torch.uint8)
            global_mask[indices] = 1

    summary = torch.tensor(
        [nonzero_count, sampled_count], dtype=torch.long, device="cuda"
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.broadcast(summary, src=0, group=utils.DEFAULT_GROUP)
    if summary[1].item() <= 0:
        raise RuntimeError("Mini-Splatting sampling selected zero Gaussians")
    local_mask = _scatter_from_rank0(global_mask, counts, local_mask_template)
    return local_mask.bool(), int(summary[0].item()), int(summary[1].item())


def _source_cdf_mask(local_importance, keep_mass):
    counts = _counts(local_importance.numel())
    global_importance = _gather_to_rank0(local_importance, counts)
    local_mask_template = torch.empty(
        local_importance.shape[0], dtype=torch.uint8, device="cuda"
    )

    global_mask = None
    if utils.DEFAULT_GROUP.rank() == 0:
        if keep_mass == 1.0:
            global_mask = torch.ones_like(global_importance, dtype=torch.uint8)
        else:
            values, _ = torch.sort(global_importance.flatten() + 1e-6)
            cumulative = torch.cumsum(values, dim=0)
            split = torch.nonzero(
                cumulative / values.sum() > (1.0 - keep_mass), as_tuple=False
            ).min()
            threshold = values[split]
            global_mask = (global_importance > threshold).to(torch.uint8)
    return _scatter_from_rank0(global_mask, counts, local_mask_template).bool()


def _tile_budget_mask(
    local_importance,
    local_pressure,
    local_touches,
    excess_instances,
    max_prune_fraction,
):
    counts = _counts(local_importance.numel())
    local_stats = torch.stack(
        (local_importance, local_pressure, local_touches), dim=1
    )
    global_stats = _gather_to_rank0(local_stats, counts)
    local_mask_template = torch.empty(
        local_importance.shape[0], dtype=torch.uint8, device="cuda"
    )

    global_mask = None
    candidate_count = 0
    selected_count = 0
    covered_touches = 0
    if utils.DEFAULT_GROUP.rank() == 0:
        importance = global_stats[:, 0]
        pressure = global_stats[:, 1]
        touches = global_stats[:, 2]
        candidate_ids = torch.nonzero(
            torch.logical_and(pressure > 0, touches > 0), as_tuple=False
        ).flatten()
        candidate_count = int(candidate_ids.numel())
        if candidate_count > 0:
            positive = importance > 0
            if positive.any():
                importance_scale = importance[positive].mean()
            else:
                importance_scale = torch.ones((), device="cuda")
            relative_importance = importance / torch.clamp_min(
                importance_scale, 1e-12
            )
            priority = pressure[candidate_ids] / (
                relative_importance[candidate_ids] + 1e-6
            )
            order = torch.argsort(priority, descending=True)
            ordered_ids = candidate_ids[order]
            cumulative_touches = torch.cumsum(touches[ordered_ids], dim=0)
            target = torch.tensor(
                float(excess_instances), device="cuda", dtype=cumulative_touches.dtype
            )
            required = int(torch.searchsorted(cumulative_touches, target).item()) + 1
            max_selected = max(
                1, int(global_stats.shape[0] * max_prune_fraction)
            )
            selected_count = min(required, max_selected, candidate_count)
            selected_ids = ordered_ids[:selected_count]
            covered_touches = int(touches[selected_ids].sum().item())
            global_mask = torch.ones(
                global_stats.shape[0], dtype=torch.uint8, device="cuda"
            )
            global_mask[selected_ids] = 0

    summary = torch.tensor(
        [candidate_count, selected_count, covered_touches],
        dtype=torch.long,
        device="cuda",
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.broadcast(summary, src=0, group=utils.DEFAULT_GROUP)
    if summary[1].item() <= 0:
        raise RuntimeError(
            "Mini-Splatting found overloaded tiles but no removable Gaussians"
        )
    local_mask = _scatter_from_rank0(global_mask, counts, local_mask_template)
    return (
        local_mask.bool(),
        int(summary[0].item()),
        int(summary[1].item()),
        int(summary[2].item()),
    )


def _global_knn_dist2(local_xyz):
    counts = _counts(local_xyz.shape[0])
    global_xyz = _gather_to_rank0(local_xyz, counts)
    global_dist2 = None
    if utils.DEFAULT_GROUP.rank() == 0:
        global_dist2 = torch.clamp_min(distCUDA2(global_xyz.contiguous()), 1e-7)
    local_template = torch.empty(
        local_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    return _scatter_from_rank0(global_dist2, counts, local_template)


def _redistribute_depth_parameters(gaussians, screenspace_pkg):
    group = utils.DEFAULT_GROUP
    rank = group.rank()
    send_ids = screenspace_pkg["local_to_gpuj_camk_send_ids"]
    sizes = screenspace_pkg["gpui_to_gpuj_imgk_size"]
    local_parameters = torch.cat(
        (gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation), dim=1
    )
    send = [
        local_parameters[send_ids[renderer][0].flatten()].contiguous()
        for renderer in range(group.size())
    ]
    recv = [
        torch.empty(
            (int(sizes[owner][rank][0]), 10),
            dtype=torch.float32,
            device="cuda",
        )
        for owner in range(group.size())
    ]
    if group.size() == 1:
        recv[0].copy_(send[0])
    else:
        dist.all_to_all(recv, send, group=group)
    redistributed = torch.cat(recv, dim=0)
    return tuple(
        parameter.contiguous()
        for parameter in torch.split(redistributed, [3, 3, 4], dim=1)
    )


def _sample_depth_candidates(
    out_points,
    remaining_transmittance,
    camera,
    strategy,
    target,
    seed,
):
    group = utils.DEFAULT_GROUP
    if utils.GLOBAL_RANK in strategy.gpu_ids:
        render_rank = strategy.gpu_ids.index(utils.GLOBAL_RANK)
        min_y = strategy.division_pos[render_rank] * utils.BLOCK_Y
        max_y = min(
            strategy.division_pos[render_rank + 1] * utils.BLOCK_Y,
            camera.image_height,
        )
        weights = (1.0 - remaining_transmittance[min_y:max_y]).flatten()
        points = (
            out_points[:, min_y:max_y]
            .permute(1, 2, 0)
            .reshape(-1, 3)
        )
        colors = camera.original_image
        if colors.dtype == torch.uint8:
            colors = colors.float() / 255.0
        else:
            colors = colors.float().clamp(0.0, 1.0)
        colors = colors.permute(1, 2, 0).reshape(-1, 3)

        positive = weights > 0
        positive_ids = torch.nonzero(positive, as_tuple=False).flatten()
        local_target = min(target, positive_ids.numel())
        if local_target > 0:
            # Exponential-race keys give exact weighted sampling without replacement;
            # a global top-k only needs each rank's local top-k candidates.
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed + group.rank())
            uniform = torch.rand(
                positive_ids.numel(), device="cuda", generator=generator
            ).clamp_min_(1e-12)
            keys = -torch.log(uniform) / weights[positive_ids]
            selected = torch.topk(
                keys, local_target, largest=False, sorted=False
            ).indices
            pixel_ids = positive_ids[selected]
            candidates = torch.cat(
                (
                    keys[selected, None],
                    points[pixel_ids],
                    colors[pixel_ids],
                ),
                dim=1,
            )
        else:
            candidates = torch.empty((0, 7), dtype=torch.float32, device="cuda")
    else:
        candidates = torch.empty((0, 7), dtype=torch.float32, device="cuda")

    candidate_counts = _counts(candidates.shape[0])
    global_candidates = _gather_to_rank0(candidates, candidate_counts)
    available = 0
    selected_candidates = None
    if group.rank() == 0:
        available = global_candidates.shape[0]
        if available >= target:
            selected = torch.topk(
                global_candidates[:, 0], target, largest=False, sorted=False
            ).indices
            selected_candidates = global_candidates[selected]

    status = torch.tensor([available, target], dtype=torch.long, device="cuda")
    if group.size() > 1:
        dist.broadcast(status, src=0, group=group)
    if status[0].item() < status[1].item():
        raise RuntimeError(
            "Mini-Splatting depth reinitialization has fewer positive-probability "
            "pixels than requested samples"
        )
    return selected_candidates


def run_mini_splatting_depth_reinitialization(
    iteration,
    train_dataset,
    gaussians,
    pipe_args,
    opt_args,
    background,
    strategy_history,
):
    args = utils.get_args()
    group = utils.DEFAULT_GROUP
    views = train_dataset.cameras
    samples_per_view = int(args.mini_splatting_num_depth / len(views))
    if samples_per_view <= 0:
        raise RuntimeError("--mini_splatting_num_depth is too small for this dataset")

    sampled_points = []
    sampled_colors = []
    before = _global_count(gaussians.get_xyz.shape[0])
    for camera_index, camera in enumerate(views):
        strategies, gpuid2tasks = start_strategy_final([camera], strategy_history)
        load_camera_from_cpu_to_all_gpu([camera], strategies, gpuid2tasks)
        screenspace_pkg = distributed_preprocess3dgs_and_all2all_final(
            [camera],
            gaussians,
            pipe_args,
            background,
            batched_strategies=strategies,
            iteration=iteration,
            mode="test",
        )
        means3D, scales, rotations = _redistribute_depth_parameters(
            gaussians, screenspace_pkg
        )
        out_points, remaining_transmittance = render_mini_splatting_depth(
            screenspace_pkg,
            strategies,
            [means3D],
            [scales],
            [rotations],
        )[0]
        selected = _sample_depth_candidates(
            out_points,
            remaining_transmittance,
            camera,
            strategies[0],
            samples_per_view,
            args.mini_splatting_seed
            + iteration * len(views)
            + camera_index * group.size(),
        )
        if group.rank() == 0:
            sampled_points.append(selected[:, 1:4])
            sampled_colors.append(selected[:, 4:7])
        camera.original_image = None
        camera.unload_image()
        del (
            screenspace_pkg,
            out_points,
            remaining_transmittance,
            means3D,
            scales,
            rotations,
        )

    global_points = None
    global_colors = None
    total_points = samples_per_view * len(views)
    if group.rank() == 0:
        global_points = torch.cat(sampled_points, dim=0).contiguous()
        global_colors = torch.cat(sampled_colors, dim=0).contiguous()
        global_dist2 = torch.clamp_min(distCUDA2(global_points), 1e-7)
    else:
        global_dist2 = None

    base = total_points // group.size()
    remainder = total_points % group.size()
    shard_counts = [base + int(rank < remainder) for rank in range(group.size())]
    local_count = shard_counts[group.rank()]
    xyz_template = torch.empty((local_count, 3), device="cuda")
    dist_template = torch.empty(local_count, device="cuda")
    local_xyz = _scatter_from_rank0(global_points, shard_counts, xyz_template)
    local_rgb = _scatter_from_rank0(global_colors, shard_counts, xyz_template)
    local_dist2 = _scatter_from_rank0(global_dist2, shard_counts, dist_template)

    gaussians.reinitialize_from_mini_splatting_depth(
        local_xyz, local_rgb, local_dist2
    )
    gaussians.training_setup(opt_args)
    train_dataset.cur_epoch_cameras = []
    _reset_strategy_history(strategy_history)
    message = (
        "Mini-Splatting depth reinitialization at iteration {}: views={} "
        "before={} after={}\n".format(
            iteration, len(views), before, total_points
        )
    )
    utils.get_log_file().write(message)
    utils.print_rank_0(message.rstrip())
    torch.cuda.empty_cache()
    return before, total_points


def _global_count(local_count):
    count = torch.tensor([local_count], dtype=torch.long, device="cuda")
    if utils.DEFAULT_GROUP.size() > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=utils.DEFAULT_GROUP)
    return int(count.item())


def _reset_strategy_history(strategy_history):
    for heuristic in strategy_history.accum_heuristic.values():
        heuristic.fill_(1.0)


def _enforce_tile_budget(
    iteration,
    stage,
    train_dataset,
    gaussians,
    pipe_args,
    opt_args,
    background,
    strategy_history,
):
    args = utils.get_args()
    budget = args.mini_splatting_tile_budget
    if budget <= 0:
        return 0

    total_removed = 0
    for round_index in range(args.mini_splatting_tile_prune_max_rounds + 1):
        importance, pressure, touches, summary = _collect_importance(
            iteration,
            train_dataset,
            gaussians,
            pipe_args,
            background,
            strategy_history,
            tile_budget=budget,
        )
        message = (
            "Mini-Splatting tile check after {} at iteration {} round {}: "
            "budget={} total_tiles={} active={} empty={} "
            "avg_all={:.2f} avg_active={:.2f} "
            "p50={} p90={} p95={} p99={} max={} "
            "overloaded={}/{} ({:.2f}%) avg_overloaded={:.2f} "
            "excess={} intersections={}\n"
        ).format(
            stage,
            iteration,
            round_index,
            budget,
            summary["total_tiles"],
            summary["active_tiles"],
            summary["empty_tiles"],
            summary["mean_occupancy"],
            summary["mean_active_occupancy"],
            summary["p50_occupancy"],
            summary["p90_occupancy"],
            summary["p95_occupancy"],
            summary["p99_occupancy"],
            summary["max_occupancy"],
            summary["overloaded_tiles"],
            summary["active_tiles"],
            summary["overloaded_ratio"],
            summary["mean_overloaded_occupancy"],
            summary["excess_instances"],
            summary["intersections"],
        )
        utils.get_log_file().write(message)
        utils.print_rank_0(message.rstrip())

        if summary["overloaded_tiles"] == 0:
            return total_removed
        if round_index == args.mini_splatting_tile_prune_max_rounds:
            warning = (
                "Mini-Splatting tile budget stopped after {} pruning rounds; "
                "the current maximum occupancy is {}.\n"
            ).format(round_index, summary["max_occupancy"])
            utils.get_log_file().write(warning)
            utils.print_rank_0(warning.rstrip())
            return total_removed

        keep, candidates, selected, covered_touches = _tile_budget_mask(
            importance,
            pressure,
            touches,
            summary["excess_instances"],
            args.mini_splatting_tile_prune_max_fraction,
        )
        before = _global_count(gaussians.get_xyz.shape[0])
        gaussians.prune_points(~keep)
        gaussians.training_setup(opt_args)
        gaussians.redistribute_gaussians()
        gaussians.reset_blur_split_stats()
        _reset_strategy_history(strategy_history)
        after = _global_count(gaussians.get_xyz.shape[0])
        total_removed += before - after
        prune_message = (
            "Mini-Splatting tile prune after {} at iteration {} round {}: "
            "before={} after={} candidates={} selected={} covered_touches={}\n"
        ).format(
            stage,
            iteration,
            round_index + 1,
            before,
            after,
            candidates,
            selected,
            covered_touches,
        )
        utils.get_log_file().write(prune_message)
        utils.print_rank_0(prune_message.rstrip())
        torch.cuda.empty_cache()

    return total_removed


def run_mini_splatting_pruning(
    iteration,
    stage,
    train_dataset,
    gaussians,
    pipe_args,
    opt_args,
    background,
    strategy_history,
):
    args = utils.get_args()
    before = _global_count(gaussians.get_xyz.shape[0])
    importance, _, _, _ = _collect_importance(
        iteration,
        train_dataset,
        gaussians,
        pipe_args,
        background,
        strategy_history,
    )

    if stage == "sample":
        keep, nonzero_count, target = _source_sampling_mask(
            importance,
            args.mini_splatting_sampling_factor,
            args.mini_splatting_seed,
        )
        detail = "nonzero={} target={}".format(nonzero_count, target)
    elif stage == "prune":
        keep = _source_cdf_mask(
            importance, args.mini_splatting_second_keep_mass
        )
        detail = "keep_mass={:.6f}".format(args.mini_splatting_second_keep_mass)
    else:
        raise ValueError("Unknown Mini-Splatting stage: {}".format(stage))

    gaussians.prune_points(~keep)
    if stage == "sample":
        global_dist2 = _global_knn_dist2(gaussians.get_xyz.detach())
        gaussians.reinitialize_after_mini_splatting(global_dist2)
        train_dataset.cur_epoch_cameras = []
    gaussians.training_setup(opt_args)
    gaussians.redistribute_gaussians()
    gaussians.reset_blur_split_stats()
    _reset_strategy_history(strategy_history)

    tile_removed = _enforce_tile_budget(
        iteration,
        stage,
        train_dataset,
        gaussians,
        pipe_args,
        opt_args,
        background,
        strategy_history,
    )

    after = _global_count(gaussians.get_xyz.shape[0])
    if args.mini_splatting_tile_budget > 0:
        detail += " tile_removed={}".format(tile_removed)
    message = (
        "Mini-Splatting {} at iteration {}: views={} before={} after={} {}\n".format(
            stage,
            iteration,
            len(train_dataset.cameras),
            before,
            after,
            detail,
        )
    )
    utils.get_log_file().write(message)
    utils.print_rank_0(message.rstrip())
    torch.cuda.empty_cache()
    return before - after
