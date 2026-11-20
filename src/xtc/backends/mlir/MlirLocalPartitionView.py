#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Static detection of local buffers that are only a view of the local partition
of a distributed operand."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.ir import BlockArgument

from .MlirPackOperandShape import (
    _full_extents_from_backend,
    operand_axis_names_for_input,
)

if TYPE_CHECKING:
    from .MlirGraphBackend import MlirGraphBackend
    from .MlirNodeBackend import MlirNodeBackend
    from .MlirNodeScheduler import MlirNodeSchedule

__all__ = ["is_view_on_local_partition", "operand_distribution"]


def operand_distribution(
    graph_backend: MlirGraphBackend,
    node_backend: MlirNodeBackend,
    operand_idx: int,
) -> list[str] | None:
    """Distribution axes of a node operand that is a distributed function argument."""
    operands = node_backend.source_op.operands
    if operand_idx >= len(operands):
        return None
    operand = operands[operand_idx]
    if not isinstance(operand, BlockArgument):
        return None
    func: Any = graph_backend.xdsl_func
    arg_attrs = func.arg_attrs
    if arg_attrs is None or operand.index >= len(arg_attrs.data):
        return None
    dist = arg_attrs.data[operand.index].data.get("sdist.distributed")
    if not isinstance(dist, ArrayAttr):
        return None
    return [str(a.data) for a in dist.data if isinstance(a, StringAttr)]


def _step(vec: list[int], dims: list[str]) -> tuple[str, int] | None:
    nz = [(dims[j], vec[j]) for j in range(len(vec)) if vec[j] != 0]
    return nz[0] if len(nz) == 1 else None


def is_view_on_local_partition(
    schedule: MlirNodeSchedule,
    *,
    root: str,
    buffer_loop: str,
    operand_idx: int,
    tiles_sizes_by_loops: dict[str, list[int]],
    node_backend: MlirNodeBackend,
    distribution: list[str] | None,
    memory_mesh: dict[str, int],
) -> bool:
    """Whether the local buffer of ``operand_idx`` at ``buffer_loop`` stays within
    the partition owned by the local memory.

    The buffer is distributed either as a function argument (``distribution``) or
    by the latest ``distributed_buffer_at`` of the operand before ``buffer_loop``.
    """
    permutation = schedule.permutation[root]
    if buffer_loop not in permutation:
        return False
    pos = permutation.index(buffer_loop)
    # Origin: function arg, or latest distributed_buffer_at before the buffer loop
    axes = distribution
    start = 0
    for i, loop in enumerate(permutation[:pos]):
        for entry in schedule.distributed_buffers.get(loop, []):
            if entry["input_idx"] == operand_idx:
                axes, start = list(entry["memory_axes"]), i + 1
    if axes is None or all(a == "*" for a in axes):
        return False
    operand_dims = operand_axis_names_for_input(
        node_backend, schedule.dims, operand_idx
    )
    if operand_dims is None or len(operand_dims) != len(axes):
        return False

    extents = _full_extents_from_backend(node_backend)

    def tile(loop: str) -> tuple[str, int] | None:
        vec = tiles_sizes_by_loops.get(loop)
        return _step(vec, schedule.dims) if vec is not None else None

    # Tilings up to the origin are already applied; only later loops are walked
    for loop in permutation[:start]:
        if (t := tile(loop)) is not None:
            extents[t[0]] = min(extents[t[0]], t[1])

    for dim, axis in zip(operand_dims, axes):
        if axis == "*":
            continue
        mesh_size = memory_mesh.get(axis)
        if mesh_size is None or extents[dim] % mesh_size != 0:
            return False
        partition = extents[dim] // mesh_size  # block owned by one memory
        extent = extents[dim]
        range_ = extent  # window seen by one memory, full extent by default
        for loop in permutation[start:pos]:
            t = tile(loop)
            if t is None or t[0] != dim:
                continue
            step = t[1]
            n_iter = -(-extent // step)
            if schedule.distribution.get(loop) == axis:
                # More iterations than memories: the buffer sweeps beyond the
                # local partition
                if n_iter > mesh_size:
                    return False
                # One iteration per memory: the window shrinks to the step
                if n_iter == mesh_size:
                    range_ = step
            extent = min(extent, step)
        if range_ > partition:  # window exceeds the local partition
            return False
    return True
