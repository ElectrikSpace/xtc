#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Logical operand extents at ``pack_at`` insertion points (post ancestor tiling)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from xdsl.ir.affine import AffineDimExpr

if TYPE_CHECKING:
    from .MlirNodeBackend import MlirNodeBackend
    from .MlirNodeScheduler import MlirNodeSchedule


def _full_extents_from_backend(node_backend: MlirNodeBackend) -> dict[str, int]:
    raw = node_backend.dims
    if not isinstance(raw, dict):
        raise TypeError(
            "Expected node_backend.dims to be a dict of abstract dim → full extent"
        )
    return {str(k): int(v) for k, v in raw.items()}


def abstract_extents_after_ancestor_tiles(
    schedule: MlirNodeSchedule,
    *,
    root: str,
    pack_loop: str,
    tiles_sizes_by_loops: dict[str, list[int]],
    node_backend: MlirNodeBackend,
) -> dict[str, int]:
    """Per-abstract-dimension extents visible inside the tiled op at ``pack_loop``.

    Matches transform insertion order: strip-mining for ``pack_loop`` itself has not
    run yet; all ancestor loops listed before ``pack_loop`` in ``permutation[root]``
    have applied their ``tile_using_for`` tile sizes where defined.
    """
    permutation = schedule.permutation[root]
    extent = _full_extents_from_backend(node_backend)
    pi = permutation.index(pack_loop)
    for lb in permutation[:pi]:
        if lb not in tiles_sizes_by_loops:
            continue
        vec = tiles_sizes_by_loops[lb]
        nz = [(schedule.dims[j], vec[j]) for j in range(len(vec)) if vec[j] != 0]
        if len(nz) != 1:
            continue
        dim_name, strip = nz[0]
        extent[dim_name] = int(strip)
    return extent


def operand_axis_names_for_input(
    node_backend: MlirNodeBackend,
    schedule_dims: Sequence[str],
    input_idx: int,
) -> tuple[str, ...] | None:
    """Map ``input_idx`` memref axes to abstract iterator names using ``indexing_maps``."""
    op = node_backend.source_op
    if op.name not in ("linalg.matmul", "linalg.generic"):
        return None
    maps_prop = op.properties.get("indexing_maps")
    if maps_prop is None:
        return None
    maps = maps_prop.data
    if input_idx >= len(maps):
        return None
    affine_map_attr = maps[input_idx]
    am = affine_map_attr.data
    names: list[str] = []
    dim_list = list(schedule_dims)
    for r in am.results:
        if isinstance(r, AffineDimExpr):
            names.append(dim_list[r.position])
    if not names:
        return None
    return tuple(names)


def tiled_tensor_shape_for_input(
    schedule: MlirNodeSchedule,
    *,
    root: str,
    pack_loop: str,
    tiles_sizes_by_loops: dict[str, list[int]],
    node_backend: MlirNodeBackend,
    input_idx: int,
) -> tuple[int, ...]:
    """Tile-shaped extents for operand ``input_idx``, memref axis order."""
    axes = operand_axis_names_for_input(node_backend, schedule.dims, input_idx)
    extents = abstract_extents_after_ancestor_tiles(
        schedule,
        root=root,
        pack_loop=pack_loop,
        tiles_sizes_by_loops=tiles_sizes_by_loops,
        node_backend=node_backend,
    )
    if axes is None:
        shape = node_backend.np_inputs_spec()[input_idx]["shape"]
        return tuple(int(x) for x in shape)
    return tuple(extents[ax] for ax in axes)
