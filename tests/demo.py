import os,sys

sys.path.append('../')

from MlirImplementer import MlirImplementer

from xdsl.dialects import func,linalg
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    i32,
    f32,
    AffineMapAttr,
)
from xdsl.dialects.arith import (
    Mulf,
    Addf,
    FastMathFlagsAttr
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import TestSSAValue
from xdsl.ir import Attribute, Block, Region
from xdsl.dialects.linalg import MatmulOp

home = os.environ.get("HOME","")

i = 512
j = 128
k = 1024
elt_type = f32
vectors_size = 8

matmul = MatmulOp(
    inputs = (
        TestSSAValue(MemRefType(elt_type,(i,k))),
        TestSSAValue(MemRefType(elt_type,(k,j))),
    ),
    outputs = (TestSSAValue(MemRefType(elt_type,(i,j))),),
)

scheduler = MlirImplementer(
    mlir_install_dir=f"{home}/bin/llvm-xdsl",
    source_op = matmul,
    dims = {'i':i,'j':j,'k':k},
    parallel_dims = ['i','j'],
    reduction_dims = ['k'],
    vectors_size = vectors_size
)

scheduler.tile("i",{'i1':4})
scheduler.tile("j",{'j1':64})
scheduler.tile("k",{'k1':8})
scheduler.interchange(['i','j','k','i1','k1','j1'])
scheduler.vectorize(['j1'])
scheduler.parallelize(['i'])
scheduler.unroll({'k1':8,'j1':64})

e = scheduler.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_lowered_ir = False,
    print_assembly=False,
    color = True,
    debug = False,
)

print(e)
