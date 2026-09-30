#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Validation of MLIR schedule transformations before code generation."""

from collections import Counter

from xtc.schedules.exceptions import ScheduleValidationError

from .MlirLoopNames import make_loop_name, parent_name
from .MlirNodeScheduler import MlirNodeSchedule


def validate_node_schedule(schedule: MlirNodeSchedule) -> None:
    """Check independently extensible tile and interchange rules for one node."""
    _check_positive_tiles(schedule)
    _check_tile_bounds(schedule)
    _check_interchanges(schedule)


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
        if dim not in schedule.dim_sizes:
            raise ScheduleValidationError(
                "Problem dimension extent is unavailable.",
                node=schedule.node_name,
                dimension=dim,
            )
        extent = schedule.dim_sizes[dim]
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
                    if outer is not None and size > outer[1]:
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
