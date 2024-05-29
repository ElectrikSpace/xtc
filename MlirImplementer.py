#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
from abc import ABC
import subprocess
import numpy


import mlir
from mlir.ir import *  # TODO: segfault after TVM

from mlir.dialects import arith, builtin, func, linalg, tensor, bufferization, memref
from xdsl.ir import Operation
from xdsl.dialects.builtin import f32

from PerfectlyNestedImplementer import PerfectlyNestedImplementer
import transform


class MlirImplementer(PerfectlyNestedImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        source_op: Operation,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
        vectors_size=16,
    ):
        super().__init__(
            mlir_install_dir,
            vectors_size,
            source_op,
            dims,
            parallel_dims,
            reduction_dims,
        )

        self.ctx = Context()
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)

        if str(source_op.operands[0].type.get_element_type()) == "f32":
            self.elt_type = F32Type.get(context=self.ctx)
            self.np_elt_type = numpy.float32
        else:
            assert False

        i = self.dims[self.parallel_dims[0]]
        j = self.dims[self.parallel_dims[1]]
        k = self.dims[self.reduction_dims[0]]

        self.inputs_types = [
            MemRefType.get(
                shape=(i, k),
                element_type=self.elt_type,
                loc=self.loc,
            ),
            MemRefType.get(
                shape=(k, j),
                element_type=self.elt_type,
                loc=self.loc,
            ),
        ]

        self.outputs_types = [
            MemRefType.get(
                shape=(i, j),
                element_type=self.elt_type,
                loc=self.loc,
            )
        ]

    def build_rtclock(self):
        f64 = F64Type.get(context=self.ctx)
        with InsertionPoint.at_block_begin(self.module.body):
            frtclock = func.FuncOp(
                name="rtclock",
                type=FunctionType.get(inputs=[], results=[f64]),
                visibility="private",
                loc=self.loc,
            )
        return frtclock

    def build_printF64(self):
        f64 = F64Type.get(context=self.ctx)
        with InsertionPoint.at_block_begin(self.module.body):
            fprint = func.FuncOp(
                name="printF64",
                type=FunctionType.get(inputs=[f64], results=[]),
                visibility="private",
                loc=self.loc,
            )
        return fprint

    def build_operator(self, f_args):
        if self.source_op.name == "linalg.matmul":
            A = f_args[0]
            B = f_args[1]
            C = f_args[2]
            scal = arith.ConstantOp(self.elt_type, 0.0)
            linalg.fill(scal, outs=[C])
            op = linalg.matmul(A, B, outs=[C])
        else:
            assert False

        return op

    def payload(self):
        with InsertionPoint.at_block_begin(self.module.body), self.loc as loc:
            f = func.FuncOp(
                name=self.payload_name,
                type=FunctionType.get(
                    inputs=self.inputs_types + self.outputs_types, results=[]
                ),
            )
            entry_block = f.add_entry_block()
        with InsertionPoint(entry_block), self.loc as loc:
            self.build_operator(f.entry_block.arguments)
            func.ReturnOp([])
        return f

    def main(self, frtclock, fprint, fmatmul):
        #
        with InsertionPoint.at_block_begin(self.module.body):
            fmain = func.FuncOp(
                name="entry",
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
        with InsertionPoint(fmain.add_entry_block()), self.loc as loc:
            inputs = []
            for ity in self.inputs_types:
                scal = arith.ConstantOp(self.elt_type, numpy.random.random())
                mem = memref.AllocOp(ity, [], [])
                linalg.fill(scal, outs=[mem])
                inputs.append(mem)
            #
            callrtclock1 = func.CallOp(frtclock, [], loc=self.loc)

            for oty in self.outputs_types:
                # scal = arith.ConstantOp(self.elt_type,0.0)
                mem = memref.AllocOp(oty, [], [])
                # linalg.fill(scal,outs=[mem])
                inputs.append(mem)

            func.CallOp(fmatmul, inputs, loc=self.loc)
            callrtclock2 = func.CallOp(frtclock, [], loc=self.loc)

            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(fprint, [time], loc=self.loc)

            for i in inputs:
                memref.DeallocOp(i)

            func.ReturnOp([], loc=self.loc)

        return fmain
