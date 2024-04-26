#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
# https://github.com/xdslproject/xdsl/blob/4c9c747622d8ad5940b4e20fe34372d274c5edee/tests/dialects/test_memref.py#L220

import subprocess
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, memref, scf, affine, vector
from xdsl.dialects.builtin import (
    FloatAttr,
    IntegerAttr,
    DenseIntOrFPElementsAttr,
    ArrayAttr,
    StringAttr,
    MemRefType,
    TensorType,
    IndexType,
    i32,
    f64,
)
from xdsl.dialects.memref import (
    Alloc,
    Cast,
    Dealloc,
    Load,
    Store,
    Global,
    GetGlobal,
)
from xdsl.dialects.arith import (
    IndexCastOp,
    Constant,
)
from xdsl.ir import BlockArgument


i = 512
j = 512
k = 512
vectors_size = 4
elt_type = builtin.f64
align = 64
assert k % vectors_size == 0

mlir_install_dir = "/home/hpompougnac/bin/llvm-xdsl"

mliropt = f"{mlir_install_dir}/bin/mlir-opt"
mlirtrans = f"{mlir_install_dir}/bin/mlir-translate"
mlircpu = f"{mlir_install_dir}/bin/mlir-cpu-runner"
libmlircpu = f"{mlir_install_dir}/lib/libmlir_runner_utils.so"
libcmlircpu = f"{mlir_install_dir}/lib/libmlir_c_runner_utils.so"

mliropt_opts = [
    "--allow-unregistered-dialect",
    "--mlir-print-op-generic",
    "--convert-scf-to-cf",
    "--convert-vector-to-llvm",
    "--convert-to-llvm",
    "--canonicalize",
]
cmd_run_mlir = [
    mlircpu,
    "-e",
    "main",
    "--entry-point-result=void",
    f"-shared-libs={libmlircpu}",
    f"-shared-libs={libcmlircpu}",
]
cmd_mlir_to_llvmir = [mlirtrans, "--mlir-to-llvmir"]
global_zeros_name = "global_zeros"


def memref_matmul(i, j, k):
    res_vector_type = vector.VectorType(elt_type, [vectors_size])
    global_zeros_memref_type = MemRefType(elt_type, [vectors_size])
    global_zeros_tensor_type = TensorType(elt_type, [vectors_size])
    a_memref_type = MemRefType(elt_type, [i, j])
    b_memref_type = MemRefType(elt_type, [j, k])
    c_memref_type = MemRefType(elt_type, [i, k])

    @builtin.ModuleOp
    @Builder.implicit_region
    def module():
        global_zeros = Global.get(
            StringAttr(global_zeros_name),
            global_zeros_memref_type,
            DenseIntOrFPElementsAttr(
                [global_zeros_tensor_type, ArrayAttr([FloatAttr(0, elt_type)])]
            ),
        )

        @Builder.implicit_region((a_memref_type, b_memref_type, c_memref_type))
        def matmul(args: tuple[BlockArgument, ...]) -> None:
            a, b, out = args

            global_zeros_get = GetGlobal.get(
                global_zeros_name, global_zeros_memref_type
            )

            lit0 = arith.Constant.from_int_and_width(0, builtin.IndexType())
            lit1 = arith.Constant.from_int_and_width(1, builtin.IndexType())
            litvec = arith.Constant.from_int_and_width(
                vectors_size, builtin.IndexType()
            )
            liti = arith.Constant.from_int_and_width(i, builtin.IndexType())
            litj = arith.Constant.from_int_and_width(j, builtin.IndexType())
            litk = arith.Constant.from_int_and_width(k, builtin.IndexType())
            lit0_f = arith.Constant(FloatAttr(0.0, elt_type))

            @Builder.implicit_region((builtin.IndexType(),))
            def outer_loop(args: tuple[BlockArgument, ...]):
                (i,) = args

                @Builder.implicit_region((builtin.IndexType(),))
                def mid_loop(args: tuple[BlockArgument, ...]):
                    (j,) = args
                    vzeros = vector.Load.build(
                        operands=[global_zeros_get, [lit0]],
                        result_types=[res_vector_type],
                    )

                    @Builder.implicit_region((builtin.IndexType(), res_vector_type))
                    def inner_loop(args: tuple[BlockArgument, ...]):
                        (k, vacc) = args
                        velem_a_i_k = vector.Load.build(
                            operands=[a, [i, k]],
                            result_types=[res_vector_type],
                        )
                        velem_b_k_j = vector.Load.build(
                            operands=[b, [k, j]],
                            result_types=[res_vector_type],
                        )
                        new_acc = vector.FMA.build(
                            operands=[velem_a_i_k, velem_b_k_j, vacc],
                            result_types=[res_vector_type],
                        )

                        affine.Yield.get(new_acc)

                    inner_loop_op = affine.For.from_region(
                        [], [], [vzeros], [res_vector_type], 0, k, inner_loop
                    )
                    vector.Store.get(inner_loop_op, out, [i, j])
                    affine.Yield.get()

                affine.For.from_region([], [], [], [], 0, j, mid_loop)
                affine.Yield.get()

            affine.For.from_region([], [], [], [], 0, i, outer_loop)

            func.Return()

        func_matmul = func.FuncOp(
            "matmul",
            ((a_memref_type, b_memref_type, c_memref_type), ()),
            matmul,
        )

        func_rtclock = func.FuncOp.external("rtclock", [], [f64])
        printF64 = func.FuncOp.external("printF64", [f64], [])

        @Builder.implicit_region(())
        def main(args: tuple[BlockArgument, ...]) -> None:
            A = Alloc.get(elt_type, alignment=align, shape=[i, j])
            B = Alloc.get(elt_type, alignment=align, shape=[j, k])
            C = Alloc.get(elt_type, alignment=align, shape=[i, k])

            time1 = func.Call(func_rtclock.sym_name.data, [], [f64])
            func.Call(func_matmul.sym_name.data, [A, B, C], [])
            time2 = func.Call(func_rtclock.sym_name.data, [], [f64])

            Dealloc.get(A)
            Dealloc.get(B)
            Dealloc.get(C)

            time3 = arith.Subf(time2, time1)
            func.Call(printF64.sym_name.data, [time3], [])

            func.Return()

        func_main = func.FuncOp(
            "main",
            ((), ()),
            main,
        )

    return module


module = memref_matmul(i, j, k)
# print(str(module))

from xdsl.transforms import lower_affine
from xdsl.ir import MLContext
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms.mlir_opt import MLIROptPass


def lower_to_llvm_dialect(module):
    ctx = MLContext(True)
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    lower_affine_pass = lower_affine.LowerAffinePass()
    lower_affine_pass.apply(ctx, module)
    mliropt_pass = MLIROptPass(executable=mliropt, arguments=tuple(mliropt_opts))
    mliropt_pass.apply(ctx, module)
    matmul_mlir = str(module)
    return matmul_mlir


llvm_dialect = lower_to_llvm_dialect(module)

result = subprocess.run(
    cmd_run_mlir, input=llvm_dialect, stdout=subprocess.PIPE, text=True
)

print(result.stdout)
