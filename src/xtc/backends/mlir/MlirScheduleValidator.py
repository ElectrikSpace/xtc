#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Validation of MLIR schedule transformations before code generation."""

from collections import Counter
from typing import TYPE_CHECKING, cast

from xtc.schedules.exceptions import ScheduleValidationError

from xtc.schedules.loop_names import basename, make_loop_name, parent_name
from .MlirNodeScheduler import MlirNodeSchedule

if TYPE_CHECKING:
    from .MlirGraphBackend import MlirGraphBackend


def validate_node_schedule(schedule: MlirNodeSchedule) -> None:
    """Check independently extensible tile and interchange rules for one node."""
    _check_positive_tiles(schedule)
    _check_tile_bounds(schedule)
    _check_interchanges(schedule)
    _check_buffers_outside_subprocessor_loops(schedule)
    _check_no_reduction_distribution(schedule)


def _check_positive_tiles(schedule: MlirNodeSchedule) -> None:
    for dim, tiles in schedule.tiles.items():
        for name, size in tiles.items():
            if size <= 0:
                raise ScheduleValidationError(
                    f"Tile {name} must have a positive size, got {size}.",
                    node=schedule.node_name,
                    root=parent_name(name),
                    dimension=dim,
                )


def _check_tile_bounds(schedule: MlirNodeSchedule) -> None:
    if schedule.dim_sizes is None:
        return
    for dim, tiles in schedule.tiles.items():
        if not tiles:
            continue
        base = basename(dim)
        if base not in schedule.dim_sizes:
            raise ScheduleValidationError(
                "Problem dimension extent is unavailable.",
                node=schedule.node_name,
                dimension=dim,
            )
        if any(name in schedule.padded_tiles for name in tiles):
            continue
        extent = schedule.dim_sizes[base]
        for name, size in tiles.items():
            if size > extent:
                raise ScheduleValidationError(
                    f"Tile {name} ({size}) exceeds problem dimension {dim} ({extent}).",
                    node=schedule.node_name,
                    root=parent_name(name),
                    dimension=dim,
                )


def _check_interchanges(schedule: MlirNodeSchedule) -> None:
    if not schedule.permutation:
        raise ScheduleValidationError(
            "No interchange is defined.", node=schedule.node_name
        )

    tiles_by_root: dict[str, set[str]] = {}
    for tiles in schedule.tiles.values():
        for tile in tiles:
            tiles_by_root.setdefault(parent_name(tile), set()).add(tile)

    splits_by_root: dict[str, dict[str, str]] = {}
    for dim, segments in schedule.splits.items():
        for segment in segments:
            splits_by_root.setdefault(parent_name(segment), {})[segment] = dim

    root = next(iter(schedule.permutation))
    child_roots = {name for segments in schedule.splits.values() for name in segments}
    for loc_root in schedule.permutation:
        if loc_root != root and loc_root not in child_roots:
            raise ScheduleValidationError(
                "Unknown split root.", node=schedule.node_name, root=loc_root
            )
    for loc_root in tiles_by_root.keys() | splits_by_root.keys() | child_roots:
        if loc_root not in schedule.permutation:
            raise ScheduleValidationError(
                "Missing interchange.", node=schedule.node_name, root=loc_root
            )

    def check_root(
        loc_root: str,
        seen_dims: set[str],
        outer_tiles: dict[str, tuple[str, int]],
    ) -> None:
        permutation = schedule.permutation[loc_root]
        counts = Counter(permutation)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ScheduleValidationError(
                f"Duplicate interchange axes {duplicates}.",
                node=schedule.node_name,
                root=loc_root,
            )

        split_axes = splits_by_root.get(loc_root, {})
        split_dims = set(split_axes.values())
        base_axes = {
            make_loop_name(loc_root, dim)
            for dim in schedule.dims
            if dim not in split_dims
        }
        required = tiles_by_root.get(loc_root, set()) | split_axes.keys()
        actual = set(permutation)
        missing = sorted(required - actual)
        unknown = sorted(actual - required - base_axes)
        if missing or unknown:
            raise ScheduleValidationError(
                f"Interchange missing axes {missing}; unknown axes {unknown}.",
                node=schedule.node_name,
                root=loc_root,
            )

        path_dims = seen_dims.copy()
        path_tiles = outer_tiles.copy()
        for axis in permutation:
            if axis in split_axes:
                path_dims.add(split_axes[axis])
                check_root(axis, path_dims, path_tiles)
            elif axis in base_axes:
                path_dims.add(axis.rsplit("/", 1)[-1])
            else:
                for dim, tiles in schedule.tiles.items():
                    if axis not in tiles:
                        continue
                    size = tiles[axis]
                    outer = path_tiles.get(dim)
                    if (
                        outer is not None
                        and size > outer[1]
                        and outer[0] not in schedule.padded_tiles
                    ):
                        raise ScheduleValidationError(
                            f"Inner tile {axis} ({size}) exceeds outer tile "
                            f"{outer[0]} ({outer[1]}).",
                            node=schedule.node_name,
                            root=loc_root,
                            dimension=dim,
                        )
                    path_tiles[dim] = (axis, size)
                    break

        if not split_axes:
            missing_dims = sorted(set(schedule.dims) - path_dims)
            if missing_dims:
                raise ScheduleValidationError(
                    f"Interchange missing base dimensions {missing_dims}.",
                    node=schedule.node_name,
                    root=loc_root,
                )

    check_root(root, set(), {})


def _check_buffers_outside_subprocessor_loops(schedule: MlirNodeSchedule) -> None:
    """Reject buffers (sdist.read/write) nested in a subprocessor-distributed loop.

    A processor axis beyond the memory mesh rank is a subprocessor axis.
    """
    if not schedule.distribution or not schedule.permutation:
        return
    proc_axes = list(schedule.processor_mesh)
    memory_rank = len(schedule.memory_mesh)
    buffered = (
        set(schedule.packed_buffers)
        | set(schedule.write_buffers)
        | set(schedule.distributed_buffers)
        | set(schedule.read_rotates)
    )

    def is_subprocessor(axis: str) -> bool:
        proc_axis = schedule.distribution.get(axis)
        if proc_axis is None or proc_axis not in proc_axes:
            return False
        return proc_axes.index(proc_axis) >= memory_rank

    def check_root(loc_root: str, inside: str | None) -> None:
        for axis in schedule.permutation.get(loc_root, []):
            if axis in buffered and inside is not None:
                raise ScheduleValidationError(
                    f"Buffer at {axis} is nested in loop {inside} distributed on "
                    f"subprocessor axis {schedule.distribution[inside]}; "
                    "sdist.read/write are not allowed there.",
                    node=schedule.node_name,
                    root=loc_root,
                )
            if inside is None and is_subprocessor(axis):
                inside = axis
            if axis in schedule.permutation and axis != loc_root:
                check_root(axis, inside)

    check_root(next(iter(schedule.permutation)), None)


def _check_no_reduction_distribution(schedule: MlirNodeSchedule) -> None:
    """Reject loops over a reduction dimension distributed on processors."""
    if not schedule.distribution or not schedule.reduction_dims:
        return
    axis_dims = {
        name: basename(dim)
        for dim, names in (*schedule.tiles.items(), *schedule.splits.items())
        for name in names
    }
    for axis, proc_axis in schedule.distribution.items():
        dim = axis_dims.get(axis) or basename(axis)
        if dim in schedule.reduction_dims:
            raise ScheduleValidationError(
                f"Distribution of reduction axis {axis} (dimension {dim}) on "
                f"processor axis {proc_axis} is not supported yet.",
                node=schedule.node_name,
                root=parent_name(axis),
                dimension=dim,
            )


def validate_distributed_buffers(
    schedule: MlirNodeSchedule, graph_backend: "MlirGraphBackend"
) -> None:
    """Reject distributed operands that are never buffered locally.

    A distributed operand (function argument with a distribution, or result of
    distributed_buffer_at) must be accessed through a pack_at/buffer_at placed
    after its distribution point.
    """
    from .MlirNodeBackend import MlirNodeBackend
    from .MlirLocalPartitionView import operand_distribution

    if not schedule.permutation:
        return
    node_backend = cast(MlirNodeBackend, graph_backend.nodes[schedule.node_name])
    # Inputs may be scalars (e.g. linalg.fill), so count operands, not tensor specs
    # Initialisers only write their own partition of the output
    if node_backend.source_op.name == "linalg.fill":
        return
    output_idx = len(node_backend.source_op.inputs)  # type: ignore
    # Loops (in order) where each operand is buffered locally
    buffered: dict[int, set[str]] = {}
    for loop, entries in schedule.packed_buffers.items():
        for input_idx, _, _ in entries:
            buffered.setdefault(input_idx, set()).add(loop)
    for loop in schedule.write_buffers:
        buffered.setdefault(output_idx, set()).add(loop)

    # Operand -> loops where it is distributed (None: function argument)
    origins: dict[int, list[str | None]] = {}
    for idx in range(output_idx + 1):
        if operand_distribution(graph_backend, node_backend, idx) is not None:
            origins.setdefault(idx, []).append(None)
    for loop, entries in schedule.distributed_buffers.items():
        for entry in entries:
            origins.setdefault(entry["input_idx"], []).append(loop)
    if not origins:
        return

    for idx, loops in origins.items():
        for origin in loops:
            if _has_inner_buffer(schedule, buffered.get(idx, set()), origin):
                continue
            where = "" if origin is None else f" at {origin}"
            kind = "buffer_at" if idx == output_idx else "pack_at"
            raise ScheduleValidationError(
                f"Distributed operand {idx}{where} has no inner {kind}; "
                "a distributed buffer must be copied to a local buffer "
                "(pack_at/buffer_at) inside its distribution point.",
                node=schedule.node_name,
            )


def _has_inner_buffer(
    schedule: MlirNodeSchedule, buffer_loops: set[str], origin: str | None
) -> bool:
    for perm in schedule.permutation.values():
        if origin is not None and origin not in perm:
            continue
        # Loops strictly after the distribution point (whole nest for arguments)
        after = perm if origin is None else perm[perm.index(origin) + 1 :]
        if buffer_loops & set(after):
            return True
    return False
