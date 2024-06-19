#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# {"fastmath" = #arith.fastmath<contract>}

from xdsl.dialects import func, linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.dialects.arith import Mulf, Addf
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region


def dim(i):
    return AffineExpr.dimension(i)


d0 = 2
d1 = 3
d2 = 4
d3 = 2
d4 = 3
d5 = 2
d6 = 2

elt_type = f32

block = Block(arg_types=(elt_type, elt_type, elt_type))
mulf = Mulf(block.args[0], block.args[1])
addf = Addf(block.args[2], mulf.results[0])
block.add_ops([mulf, addf, linalg.YieldOp(addf.results[0])])


op = linalg.Generic(
    (
        TestSSAValue(MemRefType(elt_type, [d0, d1 + d5, d2 + d6, d3])),
        TestSSAValue(MemRefType(elt_type, [d5, d6, d3, d4])),
    ),
    (TestSSAValue(MemRefType(elt_type, [d0, d1, d2, d3, d4])),),
    Region(block),
    (
        AffineMapAttr(
            AffineMap(7, 0, (dim(0), dim(1) + dim(5), dim(2) + dim(6), dim(3)))
        ),
        AffineMapAttr(AffineMap(7, 0, (dim(5), dim(6), dim(3), dim(4)))),
        AffineMapAttr(AffineMap(7, 0, (dim(0), dim(1), dim(2), dim(3), dim(4)))),
    ),
    (
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.parallel(),
        linalg.IteratorTypeAttr.reduction(),
        linalg.IteratorTypeAttr.reduction(),
    ),
)
print(str(op))
