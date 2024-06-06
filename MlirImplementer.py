#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
from abc import ABC
import subprocess
import numpy


import mlir
from mlir.ir import (
    Module,
    InsertionPoint,
    Context,
    Location,
    F32Type,
    F64Type,
    MemRefType,
    FunctionType,
    UnitAttr,
)

from mlir.dialects import arith, builtin, func, linalg, tensor, bufferization, memref
from xdsl.ir import Operation
from xdsl.dialects.builtin import f32

from xdsl.ir import Block as xdslBlock
from xdsl.ir import Region as xdslRegion
from xdsl.dialects import func as xdslfunc

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

        if str(source_op.operands[0].type.get_element_type()) == "f32":
            self.elt_type = F32Type.get(context=self.ctx)
        else:
            assert False

        self.inputs_types = [
            MemRefType.parse(str(i.type), context=self.ctx)
            for i in self.source_op.inputs
        ]
        self.outputs_types = [
            MemRefType.parse(str(i.type), context=self.ctx)
            for i in self.source_op.outputs
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

    def xdsl_operator_to_function(self):
        # Fetch data
        operands = self.source_op.operands
        operands_types = [o.type for o in operands]
        #
        payload = xdslBlock(arg_types=operands_types)
        concrete_operands = list(payload.args)
        value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

        new_op = self.source_op.clone(value_mapper=value_mapper)
        payload.add_ops([new_op, xdslfunc.Return()])
        payload_func = xdslfunc.FuncOp.from_region(
            self.payload_name, operands_types, [], xdslRegion(payload)
        )
        return payload_func

    def payload(self):
        xdsl_func = self.xdsl_operator_to_function()
        mlir_func = func.FuncOp.parse(str(xdsl_func), context=self.ctx)
        entry_block = mlir_func.regions[0].blocks[0]
        outputs = entry_block.arguments[len(self.source_op.inputs) :]
        with InsertionPoint.at_block_begin(entry_block), self.loc as loc:
            scal = arith.ConstantOp(self.elt_type, 0.0)
            for o in outputs:
                linalg.fill(scal, outs=[o])

        ip = InsertionPoint.at_block_begin(self.module.body)
        ip.insert(mlir_func)

        return mlir_func

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
                mem = memref.AllocOp(oty, [], [])
                inputs.append(mem)

            func.CallOp(fmatmul, inputs, loc=self.loc)
            callrtclock2 = func.CallOp(frtclock, [], loc=self.loc)

            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(fprint, [time], loc=self.loc)

            for i in inputs:
                memref.DeallocOp(i)

            func.ReturnOp([], loc=self.loc)

        return fmain
