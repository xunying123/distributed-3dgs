from contextlib import contextmanager
import math

import torch


_BIT_WEIGHTS = {}


@contextmanager
def _nvtx_range(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _bit_weights(device):
    key = (device.type, device.index)
    weights = _BIT_WEIGHTS.get(key)
    if weights is None:
        weights = torch.tensor(
            [1, 2, 4, 8, 16, 32, 64, 128],
            dtype=torch.int16,
            device=device,
        )
        _BIT_WEIGHTS[key] = weights
    return weights


def _pack_bitmask(mask):
    """Pack a one-dimensional boolean mask into eight rows per byte."""
    mask = mask.reshape(-1)
    num_rows = mask.numel()
    num_bytes = (num_rows + 7) // 8
    if num_bytes == 0:
        return torch.empty((0,), dtype=torch.uint8, device=mask.device)

    padded = torch.zeros((num_bytes * 8,), dtype=torch.uint8, device=mask.device)
    padded[:num_rows] = mask.to(torch.uint8)
    return (
        (padded.view(-1, 8).to(torch.int16) * _bit_weights(mask.device))
        .sum(dim=1)
        .to(torch.uint8)
    )


def _unpack_bitmask(packed, num_rows):
    """Unpack a byte bitmask and discard padding bits."""
    if num_rows == 0:
        return torch.empty((0,), dtype=torch.bool, device=packed.device)

    unpacked = packed.to(torch.int16).unsqueeze(1).bitwise_and(
        _bit_weights(packed.device).unsqueeze(0)
    )
    return unpacked.reshape(-1)[:num_rows].ne(0)


class _BitmaskSparseAllToAll(torch.autograd.Function):
    """Dense forward all-to-all with lossless row-sparse backward communication."""

    @staticmethod
    def forward(ctx, group, nvtx_name, output_shapes, *input_tensors):
        world_size = torch.distributed.get_world_size(group)
        if len(input_tensors) != world_size or len(output_shapes) != world_size:
            raise ValueError("all-to-all requires one input and one output per rank")
        if not input_tensors:
            raise ValueError("all-to-all input list cannot be empty")

        reference = input_tensors[0]
        feature_shape = tuple(reference.shape[1:])
        row_width = math.prod(feature_shape)
        for tensor in input_tensors:
            if tensor.dtype != reference.dtype or tensor.device != reference.device:
                raise ValueError("all-to-all tensors must share dtype and device")
            if tuple(tensor.shape[1:]) != feature_shape:
                raise ValueError("all-to-all tensors must share their trailing shape")

        normalized_output_shapes = tuple(tuple(shape) for shape in output_shapes)
        for shape in normalized_output_shapes:
            if math.prod(shape[1:]) != row_width:
                raise ValueError(
                    "input and output rows must contain the same number of values: "
                    f"input={tuple(reference.shape)}, output={shape}"
                )

        if world_size == 1:
            output_tensors = [input_tensors[0].reshape(normalized_output_shapes[0]).clone()]
        else:
            output_tensors = [
                torch.empty(shape, dtype=reference.dtype, device=reference.device)
                for shape in normalized_output_shapes
            ]
            torch.distributed.all_to_all(
                output_tensor_list=output_tensors,
                input_tensor_list=list(input_tensors),
                group=group,
            )

        ctx.group = group
        ctx.nvtx_name = nvtx_name
        ctx.input_shapes = tuple(tuple(tensor.shape) for tensor in input_tensors)
        ctx.output_shapes = normalized_output_shapes
        ctx.dtype = reference.dtype
        ctx.device = reference.device
        return tuple(output_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        world_size = len(ctx.input_shapes)
        if world_size == 1:
            grad = grad_outputs[0]
            if grad is None:
                grad_input = torch.zeros(
                    ctx.input_shapes[0], dtype=ctx.dtype, device=ctx.device
                )
            else:
                grad_input = grad.reshape(ctx.input_shapes[0])
            return (None, None, None, grad_input)

        rank = torch.distributed.get_rank(ctx.group)
        materialized_grads = []
        with _nvtx_range(f"backward.{ctx.nvtx_name}.bitmask"):
            for grad, shape in zip(grad_outputs, ctx.output_shapes):
                if grad is None:
                    grad = torch.zeros(shape, dtype=ctx.dtype, device=ctx.device)
                materialized_grads.append(grad.contiguous())

            # output[peer] came from that source rank. Its mask and active values
            # therefore travel back to the same peer during backward.
            output_masks = []
            packed_output_masks = []
            for peer, grad in enumerate(materialized_grads):
                if peer == rank:
                    mask = torch.empty((0,), dtype=torch.bool, device=ctx.device)
                    packed = torch.empty((0,), dtype=torch.uint8, device=ctx.device)
                else:
                    mask = (
                        grad.reshape(grad.shape[0], math.prod(grad.shape[1:]))
                        .ne(0)
                        .any(dim=1)
                    )
                    packed = _pack_bitmask(mask)
                output_masks.append(mask)
                packed_output_masks.append(packed)

            empty_mask = torch.empty((0,), dtype=torch.uint8, device=ctx.device)
            masks_to_sources = []
            masks_from_destinations = []
            for peer in range(world_size):
                if peer == rank:
                    masks_to_sources.append(empty_mask)
                    masks_from_destinations.append(empty_mask)
                else:
                    masks_to_sources.append(packed_output_masks[peer])
                    masks_from_destinations.append(
                        torch.empty(
                            ((ctx.input_shapes[peer][0] + 7) // 8,),
                            dtype=torch.uint8,
                            device=ctx.device,
                        )
                    )

            torch.distributed.all_to_all(
                output_tensor_list=masks_from_destinations,
                input_tensor_list=masks_to_sources,
                group=ctx.group,
            )

            input_masks = []
            for peer in range(world_size):
                if peer == rank:
                    input_masks.append(output_masks[rank])
                else:
                    input_masks.append(
                        _unpack_bitmask(
                            masks_from_destinations[peer], ctx.input_shapes[peer][0]
                        )
                    )

        with _nvtx_range(f"backward.{ctx.nvtx_name}.values"):
            remote_peers = [peer for peer in range(world_size) if peer != rank]
            remote_active_counts = (
                torch.stack([input_masks[peer].count_nonzero() for peer in remote_peers])
                .cpu()
                .tolist()
            )
            input_active_counts = [0] * world_size
            for peer, count in zip(remote_peers, remote_active_counts):
                input_active_counts[peer] = count
            empty_values = torch.empty(
                (0,) + tuple(ctx.input_shapes[0][1:]),
                dtype=ctx.dtype,
                device=ctx.device,
            )
            values_to_sources = []
            values_from_destinations = []
            for peer in range(world_size):
                if peer == rank:
                    values_to_sources.append(empty_values)
                    values_from_destinations.append(empty_values)
                else:
                    values_to_sources.append(
                        materialized_grads[peer][output_masks[peer]].contiguous()
                    )
                    values_from_destinations.append(
                        torch.empty(
                            (input_active_counts[peer],)
                            + tuple(ctx.input_shapes[peer][1:]),
                            dtype=ctx.dtype,
                            device=ctx.device,
                        )
                    )

            torch.distributed.all_to_all(
                output_tensor_list=values_from_destinations,
                input_tensor_list=values_to_sources,
                group=ctx.group,
            )

            grad_inputs = []
            for peer, shape in enumerate(ctx.input_shapes):
                grad_input = torch.zeros(shape, dtype=ctx.dtype, device=ctx.device)
                if peer == rank:
                    grad_input.copy_(materialized_grads[rank].reshape(shape))
                else:
                    grad_input[input_masks[peer]] = values_from_destinations[peer]
                grad_inputs.append(grad_input)

        return (None, None, None, *grad_inputs)


def bitmask_sparse_all_to_all(
    output_tensor_list, input_tensor_list, group, nvtx_name="all_to_all"
):
    """Run a dense forward and a fixed, lossless bitmask-sparse backward."""
    output_shapes = tuple(tuple(tensor.shape) for tensor in output_tensor_list)
    outputs = _BitmaskSparseAllToAll.apply(
        group, nvtx_name, output_shapes, *input_tensor_list
    )
    return list(outputs)
