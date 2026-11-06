#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph

__all__ = [
    "ggml_matmul_graph",
]


def ggml_matmul_graph(
    m: int, n: int, k: int, dtype: str = "float32", name: str = "matmul"
) -> XTCGraph:
    """XTC graph of a ggml ``MUL_MAT`` node (2D slice).

    ggml computes ``dst[n, m] = sum_k src0[k, n] * src1[k, m]`` where ne0 is the
    contiguous dimension. Seen from XTC (row-major):

    - ``A`` = src1, shape (m, k), natural layout.
    - ``B`` = src0, logical shape (k, n) but stored transposed, i.e. physically
      (n, k): ``layout=[1, 0]``.
    - ``C`` = dst, shape (m, n).

    The compiled kernel takes bare pointers ``(A=src1, B=src0, C=dst)``.
    """
    a = O.tensor((m, k), dtype, name="A")
    b = O.tensor((k, n), dtype, name="B", layout=[1, 0])
    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C")
    return gb.graph
