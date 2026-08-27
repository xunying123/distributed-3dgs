"""Distributed implementation of Mini-Splatting's simplification stages."""

import time

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
    accum_weights=None,
    projected_area=None,
    secondary_importance=None,
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
    if accum_weights is not None:
        accum_weights.add_(owner_stats[:, 0])
    if projected_area is not None:
        projected_area.add_(owner_stats[:, 1])
    if metric == "outdoor":
        valid = torch.logical_and(view_has_max, owner_stats[:, 1] != 0)
        importance[valid] += owner_stats[valid, 0] / owner_stats[valid, 1]
    else:
        importance.add_(owner_stats[:, 0])
    if secondary_importance is not None:
        secondary_valid = torch.logical_and(
            ~view_has_max, owner_stats[:, 1] != 0
        )
        secondary_importance[secondary_valid] += (
            owner_stats[secondary_valid, 0] / owner_stats[secondary_valid, 1]
        )
    overload_pressure.add_(owner_stats[:, 3])
    overloaded_touches.add_(owner_stats[:, 4])


def _log_contribution_stats(
    iteration,
    tile_budget,
    gaussians,
    importance,
    accum_weights,
    projected_area,
    area_max,
    overload_pressure,
    overloaded_touches,
    secondary_importance,
):
    """Log global primary/secondary/invisible contribution diagnostics."""
    primary = area_max > 0
    secondary = torch.logical_and(~primary, accum_weights > 0)
    invisible = torch.logical_not(torch.logical_or(primary, secondary))
    classes = (
        ("primary", primary),
        ("secondary", secondary),
        ("invisible", invisible),
    )

    opacity = gaussians.get_opacity.detach().flatten()
    mean_scale = gaussians.get_scaling.detach().mean(dim=1)
    xyz = gaussians.get_xyz.detach()
    counts = torch.tensor(
        [int(mask.sum().item()) for _, mask in classes],
        dtype=torch.long,
        device="cuda",
    )
    hotspot = torch.logical_and(overload_pressure > 0, overloaded_touches > 0)
    hotspot_counts = torch.tensor(
        [int(torch.logical_and(mask, hotspot).sum().item()) for _, mask in classes],
        dtype=torch.long,
        device="cuda",
    )
    sums = torch.zeros((3, 9), dtype=torch.float64, device="cuda")
    opacity_min = torch.full((3,), float("inf"), dtype=torch.float64, device="cuda")
    opacity_max = torch.full((3,), float("-inf"), dtype=torch.float64, device="cuda")
    xyz_min = torch.full((3, 3), float("inf"), dtype=torch.float64, device="cuda")
    xyz_max = torch.full((3, 3), float("-inf"), dtype=torch.float64, device="cuda")
    for class_id, (_, mask) in enumerate(classes):
        if not mask.any():
            continue
        sums[class_id, 0] = accum_weights[mask].double().sum()
        sums[class_id, 1] = projected_area[mask].double().sum()
        sums[class_id, 2] = area_max[mask].double().sum()
        sums[class_id, 3] = importance[mask].double().sum()
        sums[class_id, 4] = overload_pressure[mask].double().sum()
        sums[class_id, 5] = overloaded_touches[mask].double().sum()
        sums[class_id, 6] = opacity[mask].double().sum()
        sums[class_id, 7] = mean_scale[mask].double().sum()
        sums[class_id, 8] = secondary_importance[mask].double().sum()
        opacity_min[class_id] = opacity[mask].double().min()
        opacity_max[class_id] = opacity[mask].double().max()
        xyz_values = xyz[mask].double()
        xyz_min[class_id] = xyz_values.amin(dim=0)
        xyz_max[class_id] = xyz_values.amax(dim=0)

    group = utils.DEFAULT_GROUP
    if group.size() > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(hotspot_counts, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(opacity_min, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(opacity_max, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(xyz_min, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(xyz_max, op=dist.ReduceOp.MAX, group=group)

    total = int(counts.sum().item())
    lines = [
        "Mini-Splatting contribution stats at iteration {} tile_budget={}: total={}".format(
            iteration, tile_budget, total
        )
    ]
    for class_id, (name, _) in enumerate(classes):
        count = int(counts[class_id].item())
        denominator = max(count, 1)
        bbox_min = xyz_min[class_id].cpu().tolist() if count else [0.0] * 3
        bbox_max = xyz_max[class_id].cpu().tolist() if count else [0.0] * 3
        lines.append(
            "  {}: count={} ratio={:.4f}% accum_weight={:.6e} "
            "projected_area={:.0f} max_area={:.0f} importance={:.6e} "
            "secondary_score={:.6e} "
            "pressure={:.6e} touches={:.0f} hotspot_count={} "
            "hotspot_ratio={:.4f}% mean_opacity={:.6e} "
            "mean_scale={:.6e} opacity_range=[{:.6e},{:.6e}] "
            "xyz_bbox_min={} xyz_bbox_max={}".format(
                name,
                count,
                100.0 * count / max(total, 1),
                sums[class_id, 0].item(),
                sums[class_id, 1].item(),
                sums[class_id, 2].item(),
                sums[class_id, 3].item(),
                sums[class_id, 8].item(),
                sums[class_id, 4].item(),
                sums[class_id, 5].item(),
                hotspot_counts[class_id].item(),
                100.0 * hotspot_counts[class_id].item() / denominator,
                sums[class_id, 6].item() / denominator,
                sums[class_id, 7].item() / denominator,
                opacity_min[class_id].item() if count else 0.0,
                opacity_max[class_id].item() if count else 0.0,
                [round(value, 6) for value in bbox_min],
                [round(value, 6) for value in bbox_max],
            )
        )
    message = "\n".join(lines) + "\n"
    utils.get_log_file().write(message)
    utils.get_log_file().flush()
    utils.print_rank_0(message.rstrip())


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
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    importance = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    area_max = torch.zeros_like(importance)
    overload_pressure = torch.zeros_like(importance)
    overloaded_touches = torch.zeros_like(importance)
    collect_contribution_stats = args.mini_splatting_log_contribution_stats
    collect_secondary = (
        collect_contribution_stats
        or args.mini_splatting_secondary_weight > 0.0
    )
    accum_weights = torch.zeros_like(importance) if collect_contribution_stats else None
    projected_area = torch.zeros_like(importance) if collect_contribution_stats else None
    secondary_importance = (
        torch.zeros_like(importance) if collect_secondary else None
    )
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
            accum_weights,
            projected_area,
            secondary_importance,
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

    torch.cuda.synchronize()
    scan_seconds = time.perf_counter() - started_at

    if args.mini_splatting_secondary_weight > 0.0:
        importance.add_(
            secondary_importance * args.mini_splatting_secondary_weight
        )
    if collect_contribution_stats:
        _log_contribution_stats(
            iteration,
            tile_budget,
            gaussians,
            importance,
            accum_weights,
            projected_area,
            area_max,
            overload_pressure,
            overloaded_touches,
            secondary_importance,
        )
    if args.mini_splatting_secondary_weight == 0.0:
        # Preserve the repository's existing primary-contributor compatibility
        # semantics unless the secondary score is explicitly enabled.
        importance[area_max == 0] = 0
    else:
        invisible = torch.logical_and(
            area_max == 0, secondary_importance == 0
        )
        importance[invisible] = 0
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
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started_at
    collection_profile = torch.tensor(
        [
            total_seconds,
            scan_seconds,
            max(total_seconds - scan_seconds, 0.0),
            torch.cuda.memory_allocated() / (1024**3),
            torch.cuda.memory_reserved() / (1024**3),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.all_reduce(
            collection_profile,
            op=dist.ReduceOp.MAX,
            group=utils.DEFAULT_GROUP,
        )
    profile_message = (
        "Mini-Splatting importance collection at iteration {}: views={} "
        "tile_budget={} max_rank_seconds={:.3f} max_rank_scan_seconds={:.3f} "
        "max_rank_postprocess_seconds={:.3f} views_per_second={:.3f} "
        "max_rank_allocated_gib={:.3f} max_rank_reserved_gib={:.3f}\n"
    ).format(
        iteration,
        len(train_dataset.cameras),
        tile_budget,
        collection_profile[0].item(),
        collection_profile[1].item(),
        collection_profile[2].item(),
        len(train_dataset.cameras) / max(collection_profile[0].item(), 1e-12),
        collection_profile[3].item(),
        collection_profile[4].item(),
    )
    utils.get_log_file().write(profile_message)
    utils.get_log_file().flush()
    utils.print_rank_0(profile_message.rstrip())
    return (
        importance,
        overload_pressure,
        overloaded_touches,
        tile_summary,
        area_max > 0,
    )


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


def _source_sampling_mask(
    local_importance,
    local_primary,
    sampling_factor,
    seed,
    preserve_primary,
):
    counts = _counts(local_importance.numel())
    global_importance = _gather_to_rank0(local_importance, counts)
    global_primary = None
    if preserve_primary:
        global_primary = _gather_to_rank0(local_primary.to(torch.uint8), counts)
    local_mask_template = torch.empty(
        local_importance.shape[0], dtype=torch.uint8, device="cuda"
    )

    global_mask = None
    sampled_count = 0
    nonzero_count = 0
    primary_count = 0
    selected_primary_count = 0
    selected_secondary_count = 0
    if utils.DEFAULT_GROUP.rank() == 0:
        positive_ids = torch.nonzero(global_importance > 0, as_tuple=False).flatten()
        nonzero_count = int(positive_ids.numel())
        sampled_count = int(nonzero_count * sampling_factor)
        if sampled_count > 0:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            if preserve_primary:
                positive_primary = global_primary[positive_ids].bool()
                primary_ids = positive_ids[positive_primary]
                secondary_ids = positive_ids[~positive_primary]
                primary_count = int(primary_ids.numel())
                if sampled_count >= primary_count:
                    selected_primary_ids = primary_ids
                    secondary_target = sampled_count - primary_count
                    if secondary_target > 0:
                        secondary_sampled_ids = (
                            _weighted_sample_without_replacement(
                                global_importance[secondary_ids],
                                secondary_target,
                                generator=generator,
                            )
                        )
                        selected_secondary_ids = secondary_ids[
                            secondary_sampled_ids
                        ]
                    else:
                        selected_secondary_ids = secondary_ids[:0]
                else:
                    primary_sampled_ids = _weighted_sample_without_replacement(
                        global_importance[primary_ids],
                        sampled_count,
                        generator=generator,
                    )
                    selected_primary_ids = primary_ids[primary_sampled_ids]
                    selected_secondary_ids = secondary_ids[:0]
                selected_primary_count = int(selected_primary_ids.numel())
                selected_secondary_count = int(selected_secondary_ids.numel())
                indices = torch.cat(
                    (selected_primary_ids, selected_secondary_ids), dim=0
                )
            else:
                sampled_ids = _weighted_sample_without_replacement(
                    global_importance[positive_ids],
                    sampled_count,
                    generator=generator,
                )
                indices = positive_ids[sampled_ids]
            global_mask = torch.zeros_like(global_importance, dtype=torch.uint8)
            global_mask[indices] = 1

    summary = torch.tensor(
        [
            nonzero_count,
            sampled_count,
            primary_count,
            selected_primary_count,
            selected_secondary_count,
        ],
        dtype=torch.long,
        device="cuda",
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.broadcast(summary, src=0, group=utils.DEFAULT_GROUP)
    if summary[1].item() <= 0:
        raise RuntimeError("Mini-Splatting sampling selected zero Gaussians")
    local_mask = _scatter_from_rank0(global_mask, counts, local_mask_template)
    return (
        local_mask.bool(),
        int(summary[0].item()),
        int(summary[1].item()),
        int(summary[2].item()),
        int(summary[3].item()),
        int(summary[4].item()),
    )


def _source_cdf_mask(local_importance, keep_mass):
    counts = _counts(local_importance.numel())
    global_importance = _gather_to_rank0(local_importance, counts)
    local_mask_template = torch.empty(
        local_importance.shape[0], dtype=torch.uint8, device="cuda"
    )

    global_mask = None
    keep_count = 0
    total_mass = 0.0
    retained_mass = 0.0
    threshold = 0.0
    if utils.DEFAULT_GROUP.rank() == 0:
        nonnegative_importance = torch.clamp_min(global_importance.flatten(), 0)
        total_mass = float(nonnegative_importance.double().sum().item())
        if keep_mass == 1.0:
            global_mask = torch.ones_like(global_importance, dtype=torch.uint8)
            keep_count = global_importance.numel()
            retained_mass = total_mass
            threshold = float(nonnegative_importance.min().item())
        else:
            if total_mass <= 0.0:
                raise RuntimeError(
                    "Mini-Splatting CDF pruning requires positive importance mass"
                )
            values, indices = torch.sort(
                nonnegative_importance, descending=True
            )
            cumulative = torch.cumsum(values.double(), dim=0)
            target_mass = total_mass * keep_mass
            keep_count = min(
                int(torch.searchsorted(cumulative, target_mass).item()) + 1,
                values.numel(),
            )
            selected = indices[:keep_count]
            global_mask = torch.zeros_like(global_importance, dtype=torch.uint8)
            global_mask[selected] = 1
            retained_mass = float(cumulative[keep_count - 1].item())
            threshold = float(values[keep_count - 1].item())

    summary = torch.tensor(
        [keep_count, total_mass, retained_mass, threshold],
        dtype=torch.float64,
        device="cuda",
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.broadcast(summary, src=0, group=utils.DEFAULT_GROUP)
    local_mask = _scatter_from_rank0(
        global_mask, counts, local_mask_template
    ).bool()
    return (
        local_mask,
        int(summary[0].item()),
        summary[2].item() / max(summary[1].item(), 1e-30),
        summary[3].item(),
    )


def _tile_budget_mask(
    local_importance,
    local_pressure,
    local_touches,
    local_primary,
    excess_instances,
    max_prune_fraction,
    pressure_exponent,
    touch_exponent,
    primary_penalty,
    preserve_primary,
):
    counts = _counts(local_importance.numel())
    local_stats = torch.stack(
        (
            local_importance,
            local_pressure,
            local_touches,
            local_primary.to(local_importance.dtype),
        ),
        dim=1,
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
        candidate_mask = torch.logical_and(pressure > 0, touches > 0)
        if preserve_primary:
            candidate_mask.logical_and_(global_stats[:, 3] == 0)
        candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        candidate_count = int(candidate_ids.numel())
        global_mask = torch.ones(
            global_stats.shape[0], dtype=torch.uint8, device="cuda"
        )
        if candidate_count > 0:
            positive = importance > 0
            if positive.any():
                importance_scale = importance[positive].mean()
            else:
                importance_scale = torch.ones((), device="cuda")
            relative_importance = importance / torch.clamp_min(
                importance_scale, 1e-12
            )
            priority = pressure[candidate_ids].pow(pressure_exponent)
            if touch_exponent > 0.0:
                priority.div_(touches[candidate_ids].pow(touch_exponent))
            priority.div_(relative_importance[candidate_ids] + 1e-6)
            if primary_penalty > 1.0 and not preserve_primary:
                candidate_primary = global_stats[candidate_ids, 3] != 0
                priority[candidate_primary] = (
                    priority[candidate_primary] / primary_penalty
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
            global_mask[selected_ids] = 0

    summary = torch.tensor(
        [candidate_count, selected_count, covered_touches],
        dtype=torch.long,
        device="cuda",
    )
    if utils.DEFAULT_GROUP.size() > 1:
        dist.broadcast(summary, src=0, group=utils.DEFAULT_GROUP)
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


def _reset_pruning_accumulators(gaussians):
    """Reset non-optimizer per-Gaussian buffers after a topology change."""
    point_count = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    gaussians.xyz_gradient_accum = torch.zeros((point_count, 1), device=device)
    gaussians.denom = torch.zeros((point_count, 1), device=device)
    gaussians.max_radii2D = torch.zeros(point_count, device=device)
    gaussians.sum_visible_count_in_one_batch = torch.zeros(
        point_count, device=device
    )
    gaussians.send_to_gpui_cnt = torch.zeros(
        (point_count, gaussians.group_for_redistribution().size()),
        dtype=torch.int,
        device=device,
    )
    gaussians.reset_blur_split_stats()


def _set_post_prune_learning_rates(gaussians, opt_args, scale):
    """Set post-pruning learning rates without compounding across stages."""
    gaussians.post_prune_xyz_lr_scale = scale
    if opt_args.lr_scale_mode == "linear":
        batch_scale = utils.get_args().bsz
    elif opt_args.lr_scale_mode == "sqrt":
        batch_scale = utils.get_args().bsz**0.5
    elif opt_args.lr_scale_mode == "accumu":
        batch_scale = 1.0
    else:
        raise ValueError(
            "Unsupported lr_scale_mode: {}".format(opt_args.lr_scale_mode)
        )
    base_lrs = {
        "f_dc": opt_args.feature_lr * batch_scale,
        "f_rest": opt_args.feature_lr / 20.0 * batch_scale,
        "opacity": opt_args.opacity_lr * batch_scale,
        "scaling": opt_args.scaling_lr
        * utils.get_args().lr_scale_pos_and_scale
        * batch_scale,
        "rotation": opt_args.rotation_lr * batch_scale,
    }
    for group in gaussians.optimizer.param_groups:
        if group["name"] in base_lrs:
            group["lr"] = base_lrs[group["name"]] * scale


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
        importance, pressure, touches, summary, primary_mask = _collect_importance(
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
            "overloaded={}/{} ({:.2f}%) within_budget_all={:.2f}% "
            "within_budget_active={:.2f}% avg_overloaded={:.2f} "
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
            100.0
            * (summary["total_tiles"] - summary["overloaded_tiles"])
            / max(summary["total_tiles"], 1),
            100.0 - summary["overloaded_ratio"],
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
            primary_mask,
            summary["excess_instances"],
            args.mini_splatting_tile_prune_max_fraction,
            args.mini_splatting_tile_pressure_exponent,
            args.mini_splatting_tile_touch_exponent,
            args.mini_splatting_tile_primary_penalty,
            args.mini_splatting_tile_preserve_primary,
        )
        if selected <= 0:
            warning = (
                "Mini-Splatting tile budget stopped at round {}: "
                "no removable candidates (preserve_primary={}).\n"
            ).format(round_index + 1, args.mini_splatting_tile_preserve_primary)
            utils.get_log_file().write(warning)
            utils.print_rank_0(warning.rstrip())
            return total_removed
        before = _global_count(gaussians.get_xyz.shape[0])
        selected_local = ~keep
        selected_primary = torch.logical_and(selected_local, primary_mask)
        selected_secondary = torch.logical_and(
            selected_local,
            torch.logical_and(~primary_mask, importance > 0),
        )
        selected_metrics = torch.stack(
            (
                selected_local.double().sum(),
                torch.logical_and(selected_local, importance == 0).double().sum(),
                selected_primary.double().sum(),
                selected_secondary.double().sum(),
                importance[selected_local].double().sum(),
                pressure[selected_local].double().sum(),
                touches[selected_local].double().sum(),
                gaussians.get_opacity[selected_local].double().sum(),
                gaussians.get_scaling[selected_local]
                .max(dim=1)
                .values.double()
                .sum(),
            )
        )
        if utils.DEFAULT_GROUP.size() > 1:
            dist.all_reduce(
                selected_metrics,
                op=dist.ReduceOp.SUM,
                group=utils.DEFAULT_GROUP,
            )
        selected_denominator = max(selected_metrics[0].item(), 1.0)
        torch.cuda.synchronize()
        prune_started_at = time.perf_counter()
        gaussians.prune_points(~keep)
        torch.cuda.synchronize()
        prune_seconds = time.perf_counter() - prune_started_at
        optimizer_started_at = time.perf_counter()
        if args.mini_splatting_preserve_optimizer_state:
            _reset_pruning_accumulators(gaussians)
        else:
            gaussians.training_setup(opt_args)
        torch.cuda.synchronize()
        optimizer_rebuild_seconds = time.perf_counter() - optimizer_started_at
        redistribute_started_at = time.perf_counter()
        gaussians.redistribute_gaussians()
        torch.cuda.synchronize()
        redistribute_seconds = time.perf_counter() - redistribute_started_at
        gaussians.reset_blur_split_stats()
        _reset_strategy_history(strategy_history)
        after = _global_count(gaussians.get_xyz.shape[0])
        total_removed += before - after
        round_profile = torch.tensor(
            [prune_seconds, optimizer_rebuild_seconds, redistribute_seconds],
            dtype=torch.float64,
            device="cuda",
        )
        if utils.DEFAULT_GROUP.size() > 1:
            dist.all_reduce(
                round_profile,
                op=dist.ReduceOp.MAX,
                group=utils.DEFAULT_GROUP,
            )
        prune_seconds, optimizer_rebuild_seconds, redistribute_seconds = (
            round_profile.cpu().tolist()
        )
        prune_message = (
            "Mini-Splatting tile prune after {} at iteration {} round {}: "
            "before={} after={} candidates={} selected={} covered_touches={} "
            "pressure_exponent={:.3f} touch_exponent={:.3f} "
            "primary_penalty={:.3f} preserve_primary={} "
            "selected_primary={} selected_secondary={} "
            "selected_zero_importance={} "
            "selected_importance_mean={:.6e} selected_pressure_mean={:.6e} "
            "selected_touches_mean={:.3f} selected_opacity_mean={:.6e} "
            "selected_scale_mean={:.6e} "
            "prune_seconds={:.3f} optimizer_rebuild_seconds={:.3f} "
            "redistribute_seconds={:.3f} preserve_optimizer_state={}\n"
        ).format(
            stage,
            iteration,
            round_index + 1,
            before,
            after,
            candidates,
            selected,
            covered_touches,
            args.mini_splatting_tile_pressure_exponent,
            args.mini_splatting_tile_touch_exponent,
            args.mini_splatting_tile_primary_penalty,
            args.mini_splatting_tile_preserve_primary,
            int(selected_metrics[2].item()),
            int(selected_metrics[3].item()),
            int(selected_metrics[1].item()),
            selected_metrics[4].item() / selected_denominator,
            selected_metrics[5].item() / selected_denominator,
            selected_metrics[6].item() / selected_denominator,
            selected_metrics[7].item() / selected_denominator,
            selected_metrics[8].item() / selected_denominator,
            prune_seconds,
            optimizer_rebuild_seconds,
            redistribute_seconds,
            args.mini_splatting_preserve_optimizer_state,
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
    diagnostic_tile_budget = (
        args.mini_splatting_tile_budget
        if args.mini_splatting_diagnostic_only
        else 0
    )
    (
        importance,
        pressure,
        touches,
        diagnostic_tile_summary,
        primary_mask,
    ) = _collect_importance(
        iteration,
        train_dataset,
        gaussians,
        pipe_args,
        background,
        strategy_history,
        tile_budget=diagnostic_tile_budget,
    )

    if args.mini_splatting_diagnostic_only:
        overload_totals = torch.stack(
            (pressure.double().sum(), touches.double().sum())
        )
        if utils.DEFAULT_GROUP.size() > 1:
            dist.all_reduce(
                overload_totals,
                op=dist.ReduceOp.SUM,
                group=utils.DEFAULT_GROUP,
            )
        message = (
            "Mini-Splatting diagnostic-only at iteration {}: views={} "
            "gaussians={} tile_budget={} total_tiles={} active={} empty={} "
            "avg_all={:.2f} avg_active={:.2f} p50={} p90={} p95={} p99={} "
            "max={} overloaded={} overloaded_ratio={:.2f}% "
            "within_budget_all={:.2f}% within_budget_active={:.2f}% "
            "mean_overloaded={:.2f} excess={} intersections={} "
            "pressure_sum={:.6e} touches_sum={:.0f}\n"
        ).format(
            iteration,
            len(train_dataset.cameras),
            before,
            diagnostic_tile_budget,
            diagnostic_tile_summary["total_tiles"],
            diagnostic_tile_summary["active_tiles"],
            diagnostic_tile_summary["empty_tiles"],
            diagnostic_tile_summary["mean_occupancy"],
            diagnostic_tile_summary["mean_active_occupancy"],
            diagnostic_tile_summary["p50_occupancy"],
            diagnostic_tile_summary["p90_occupancy"],
            diagnostic_tile_summary["p95_occupancy"],
            diagnostic_tile_summary["p99_occupancy"],
            diagnostic_tile_summary["max_occupancy"],
            diagnostic_tile_summary["overloaded_tiles"],
            diagnostic_tile_summary["overloaded_ratio"],
            100.0
            * (
                diagnostic_tile_summary["total_tiles"]
                - diagnostic_tile_summary["overloaded_tiles"]
            )
            / max(diagnostic_tile_summary["total_tiles"], 1),
            100.0 - diagnostic_tile_summary["overloaded_ratio"],
            diagnostic_tile_summary["mean_overloaded_occupancy"],
            diagnostic_tile_summary["excess_instances"],
            diagnostic_tile_summary["intersections"],
            overload_totals[0].item(),
            overload_totals[1].item(),
        )
        utils.get_log_file().write(message)
        utils.get_log_file().flush()
        utils.print_rank_0(message.rstrip())
        torch.cuda.empty_cache()
        return 0

    if stage == "sample":
        (
            keep,
            nonzero_count,
            target,
            primary_count,
            selected_primary,
            selected_secondary,
        ) = _source_sampling_mask(
            importance,
            primary_mask,
            args.mini_splatting_sampling_factor,
            args.mini_splatting_seed,
            args.mini_splatting_preserve_primary,
        )
        detail = (
            "nonzero={} target={} preserve_primary={} primary={} "
            "selected_primary={} selected_secondary={}"
        ).format(
            nonzero_count,
            target,
            args.mini_splatting_preserve_primary,
            primary_count,
            selected_primary,
            selected_secondary,
        )
    elif stage == "prune":
        keep, keep_count, achieved_mass, threshold = _source_cdf_mask(
            importance, args.mini_splatting_second_keep_mass
        )
        detail = (
            "keep_mass={:.6f} achieved_mass={:.8f} cdf_keep_count={} "
            "threshold={:.8e}"
        ).format(
            args.mini_splatting_second_keep_mass,
            achieved_mass,
            keep_count,
            threshold,
        )
    else:
        raise ValueError("Unknown Mini-Splatting stage: {}".format(stage))

    gaussians.prune_points(~keep)
    reinitialized = False
    if stage == "sample":
        if not args.mini_splatting_skip_sample_reinitialization:
            global_dist2 = _global_knn_dist2(gaussians.get_xyz.detach())
            gaussians.reinitialize_after_mini_splatting(global_dist2)
            reinitialized = True
        train_dataset.cur_epoch_cameras = []
    if reinitialized or not args.mini_splatting_preserve_optimizer_state:
        gaussians.training_setup(opt_args)
    else:
        _reset_pruning_accumulators(gaussians)
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
    _set_post_prune_learning_rates(
        gaussians, opt_args, args.mini_splatting_post_prune_lr_scale
    )

    after = _global_count(gaussians.get_xyz.shape[0])
    if args.mini_splatting_tile_budget > 0:
        detail += " tile_removed={}".format(tile_removed)
    detail += " secondary_weight={:.6f}".format(
        args.mini_splatting_secondary_weight
    )
    detail += " optimizer_state_preserved={}".format(
        args.mini_splatting_preserve_optimizer_state and not reinitialized
    )
    detail += " post_prune_lr_scale={:.6f}".format(
        args.mini_splatting_post_prune_lr_scale
    )
    if stage == "sample":
        detail += " reinitialized={}".format(
            not args.mini_splatting_skip_sample_reinitialization
        )
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
