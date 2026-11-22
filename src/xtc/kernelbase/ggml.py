#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any

import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.graph import XTCGraph
from xtc.quant.ggml import ggml_quant_format

__all__ = [
    "ggml_matmul_graph",
    "ggml_matmul_q_graph",
]


def ggml_matmul_graph(
    m: int,
    n: int,
    k: int,
    dtype: str = "float32",
    name: str = "matmul",
    device: Any = None,
) -> XTCGraph:
    """XTC graph of a ggml ``MUL_MAT`` node (2D slice).

    ggml computes ``dst[n, m] = sum_k src0[k, n] * src1[k, m]`` where ne0 is the
    contiguous dimension. Seen from XTC (row-major):

    - ``A`` = src1, shape (m, k), natural layout.
    - ``B`` = src0, logical shape (k, n) but stored transposed, i.e. physically
      (n, k): ``layout=[1, 0]``.
    - ``C`` = dst, shape (m, n).

    The compiled kernel takes bare pointers ``(A=src1, B=src0, C=dst)``.
    ``device`` places the tensors on an accelerator (e.g. the MPPA device).
    """
    a = O.tensor((m, k), dtype, name="A", device=device)
    b = O.tensor((k, n), dtype, name="B", layout=[1, 0], device=device)
    with O.graph(name=name) as gb:
        O.matmul(a, b, name="C", device=device)
    return gb.graph



def ggml_matmul_q_graph(
    m: int, n: int, k: int, qtype: str, name: str = "matmul_q"
) -> XTCGraph:
    """XTC graph of a ggml ``MUL_MAT`` node with a quantized src0.

    Same convention as :func:`ggml_matmul_graph`, except that ``Bq`` = src0 is
    the raw ggml buffer of ``n`` rows of ``k // qk`` blocks, i.e. a uint8
    tensor of shape (n, k // qk, block_bytes). It is decoded by the ``B``
    dequantize node into the logical (k, n) float32 operand of ``C``.

    The compiled kernel takes bare pointers ``(A=src1, Bq=src0, C=dst)``.
    """
    fmt = ggml_quant_format(qtype)
    a = O.tensor((m, k), "float32", name="A")
    bq = O.tensor(fmt.quantized_shape(n, k), "uint8", name="Bq")
    with O.graph(name=name) as gb:
        b = O.dequantize(bq, format=fmt.name, transpose=True, name="B")
        O.matmul(a, b, name="C")
    return gb.graph
