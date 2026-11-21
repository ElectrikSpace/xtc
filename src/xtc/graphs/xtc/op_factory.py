#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

from .context import XTCGraphContext
from .data import XTCTensorType
from .expr import (
    XTCExpr,
    XTCTensorExpr,
    XTCMatmulExpr,
    XTCDequantizeExpr,
    XTCReluExpr,
    XTCAddExpr,
    XTCMulExpr,
    XTCFmaExpr,
    XTCExpExpr,
    XTCRsqrtExpr,
    XTCSiluExpr,
    XTCCastExpr,
    XTCReduceMaxExpr,
    XTCReduceSumExpr,
    XTCSubRowExpr,
    XTCDivRowExpr,
    XTCConv2DExpr,
    XTCPad2DExpr,
    XTCUnpadExpr,
    XTCPadExpr,
    XTCReshapeExpr,
    XTCTransposeExpr,
)

__all__ = [
    "matmul",
    "dequantize",
    "conv2d",
    "pad2d",
    "unpad",
    "pad",
    "relu",
    "add",
    "mul",
    "fma",
    "exp",
    "rsqrt",
    "silu",
    "cast",
    "reduce_max",
    "reduce_sum",
    "sub_row",
    "div_row",
    "reshape",
    "transpose",
    "tensor",
    "inputs",
    "outputs",
    "type",
]


def matmul(a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCMatmulExpr(a, b, **attrs),
        name=name,
    )


def dequantize(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCDequantizeExpr(inp, **attrs),
        name=name,
    )


def conv2d(
    inp: XTCExpr, weight: XTCExpr, name: str | None = None, **attrs: Any
) -> XTCExpr:
    return XTCGraphContext.append(
        XTCConv2DExpr(inp, weight, **attrs),
        name=name,
    )


def pad2d(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCPad2DExpr(inp, **attrs),
        name=name,
    )


def unpad(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCUnpadExpr(inp, **attrs),
        name=name,
    )


def pad(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCPadExpr(inp, **attrs),
        name=name,
    )


def relu(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCReluExpr(inp, **attrs),
        name=name,
    )


def add(a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCAddExpr(a, b, **attrs),
        name=name,
    )


def mul(a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCMulExpr(a, b, **attrs),
        name=name,
    )


def fma(
    a: XTCExpr, b: XTCExpr, c: XTCExpr, name: str | None = None, **attrs: Any
) -> XTCExpr:
    """Fused elementwise a * b + c."""
    return XTCGraphContext.append(
        XTCFmaExpr(a, b, c, **attrs),
        name=name,
    )


def exp(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCExpExpr(inp, **attrs), name=name)


def rsqrt(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(XTCRsqrtExpr(inp, **attrs), name=name)


def silu(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    """Elementwise x * sigmoid(x)."""
    return XTCGraphContext.append(XTCSiluExpr(inp, **attrs), name=name)


def cast(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    """Elementwise conversion to the dtype attribute."""
    return XTCGraphContext.append(XTCCastExpr(inp, **attrs), name=name)


def reduce_max(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    """Max of each row of a 2D tensor: (M, N) -> (M,)."""
    return XTCGraphContext.append(XTCReduceMaxExpr(inp, **attrs), name=name)


def reduce_sum(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    """Sum of each row of a 2D tensor: (M, N) -> (M,)."""
    return XTCGraphContext.append(XTCReduceSumExpr(inp, **attrs), name=name)


def sub_row(
    a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any
) -> XTCExpr:
    """a[i, j] - b[i] for a (M, N) and b (M,)."""
    return XTCGraphContext.append(XTCSubRowExpr(a, b, **attrs), name=name)


def div_row(
    a: XTCExpr, b: XTCExpr, name: str | None = None, **attrs: Any
) -> XTCExpr:
    """a[i, j] / b[i] for a (M, N) and b (M,)."""
    return XTCGraphContext.append(XTCDivRowExpr(a, b, **attrs), name=name)


def reshape(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCReshapeExpr(inp, **attrs),
        name=name,
    )


def transpose(inp: XTCExpr, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCTransposeExpr(inp, **attrs),
        name=name,
    )


def tensor(*args: Any, name: str | None = None, **attrs: Any) -> XTCExpr:
    return XTCGraphContext.append(
        XTCTensorExpr(*args, **attrs),
        name=name,
    )


def outputs(*outs: XTCExpr) -> None:
    XTCGraphContext.outputs(*outs)


def inputs(*inps: XTCExpr) -> None:
    XTCGraphContext.inputs(*inps)


def type(*args: Any, **attrs: Any) -> XTCTensorType:
    return XTCTensorType(*args, **attrs)
