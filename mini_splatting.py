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
    render_stats, screenspace_pkg, importance, area_max, metric
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
            (local_send_ids[renderer][0].numel(), 3),
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
        (importance.shape[0], 3), dtype=torch.float32, device="cuda"
    )
    for renderer in range(group.size()):
        ids = local_send_ids[renderer][0].to(
            device="cuda", dtype=torch.long
        ).flatten()
        if ids.numel() > 0:
            owner_stats.index_add_(0, ids, recv[renderer])

    view_has_max = owner_stats[:, 2] != 0
    area_max.add_(owner_stats[:, 2])
    if metric == "outdoor":
        valid = torch.logical_and(view_has_max, owner_stats[:, 1] != 0)
        importance[valid] += owner_stats[valid, 0] / owner_stats[valid, 1]
    else:
        importance.add_(owner_stats[:, 0])


def _collect_importance(
    iteration,
    train_dataset,
    gaussians,
    pipe_args,
    background,
    strategy_history,
):
    args = utils.get_args()
    importance = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    area_max = torch.zeros_like(importance)

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
        render_stats = render_mini_splatting_importance(
            screenspace_pkg, strategies
        )[0]
        _route_stats_to_owners(
            render_stats,
            screenspace_pkg,
            importance,
            area_max,
            args.mini_splatting_imp_metric,
        )
        del render_stats, screenspace_pkg

    importance[area_max == 0] = 0
    return importance


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
        nonzero_count = int((global_importance != 0).sum().item())
        sampled_count = int(nonzero_count * sampling_factor)
        if sampled_count > 0:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            indices = torch.multinomial(
                global_importance,
                sampled_count,
                replacement=False,
                generator=generator,
            )
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
    importance = _collect_importance(
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
    _reset_strategy_history(strategy_history)

    after = _global_count(gaussians.get_xyz.shape[0])
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
