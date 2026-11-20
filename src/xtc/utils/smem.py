#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Sequence

from xtc.graphs.xtc.data import XTCTensorType

__all__ = ["graph_smem_offsets"]

_ELT_BYTES = {"float16": 2, "float32": 4, "float64": 8}


def graph_smem_offsets(types: Sequence[XTCTensorType]) -> list[int | None]:
    """Byte offset of each smem buffer, None for the other ones.

    The types must be the graph inputs followed by the graph outputs, which is
    the order of the function arguments.
    """
    if not any(t.memory == "smem" for t in types):
        return [None] * len(types)
    from mlir_sdist.extras.smem import smem_offsets

    buffers: list[tuple[list[int], list[str], int] | None] = []
    mesh: dict[str, int] = {}
    for t in types:
        if t.memory != "smem":
            buffers.append(None)
            continue
        assert t.distribution is not None and t.device is not None
        mesh = t.device.memory_mesh
        elt_bytes = _ELT_BYTES[t.constant_dtype]
        shape = list(t.constant_shape)
        if t.layout is not None:
            shape = [shape[idx] for idx in t.layout]
        buffers.append((shape, t.distribution, elt_bytes))
    return smem_offsets(buffers, mesh)
