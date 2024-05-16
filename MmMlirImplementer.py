#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess
import numpy

from mlir.ir import *
import mlir
from mlir.dialects import arith, builtin, func, linalg, tensor, bufferization

from PerfectlyNestedImplementer import PerfectlyNestedImplementer
import transform


class MmMlirImplementer(PerfectlyNestedImplementer):
    def __init__(
        self,
        mlir_install_dir: str,
        dims: dict[str, int],
        parallel_dims: list[str],
        reduction_dims: list[str],
    ):
        super().__init__(mlir_install_dir, dims, parallel_dims, reduction_dims)

        self.ctx = Context()
        self.elt_type = F32Type.get(context=self.ctx)
        self.np_elt_type = numpy.float32
        self.loc = Location.unknown(self.ctx)
        self.module = builtin.ModuleOp(loc=self.loc)

        self.i = self.dims[self.parallel_dims[0]]
        self.j = self.dims[self.parallel_dims[1]]
        self.k = self.dims[self.reduction_dims[0]]

        self.A_memref_type = MemRefType.get(
            shape=(self.i, self.k),
            element_type=self.elt_type,
            loc=self.loc,
        )
        self.B_memref_type = MemRefType.get(
            shape=(self.k, self.j),
            element_type=self.elt_type,
            loc=self.loc,
        )
        self.C_memref_type = MemRefType.get(
            shape=(self.i, self.j),
            element_type=self.elt_type,
            loc=self.loc,
        )

    def initialize_tensor(self, shape, scalar_value):
        with self.loc as loc:
            tensor_type = RankedTensorType.get(
                shape=shape,
                element_type=self.elt_type,
            )

            memref_type = MemRefType.get(
                shape=shape,
                element_type=self.elt_type,
            )

            numpy_value = numpy.full(
                shape,
                scalar_value,
                dtype=self.np_elt_type,
            )
            value = DenseElementsAttr.get(
                numpy_value,
            )
            cons = arith.ConstantOp(tensor_type, value)
            buff = bufferization.to_memref(tensor=cons, memref=memref_type)
            return buff

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

    def payload(self):
        with InsertionPoint.at_block_begin(self.module.body), self.loc as loc:
            f = func.FuncOp(
                name=self.payload_name,
                type=FunctionType.get(
                    inputs=[self.A_memref_type, self.B_memref_type, self.C_memref_type],
                    results=[],
                ),
            )
            entry_block = f.add_entry_block()
        with InsertionPoint(entry_block), self.loc as loc:
            A = f.entry_block.arguments[0]
            B = f.entry_block.arguments[1]
            C = f.entry_block.arguments[2]
            matmul = linalg.matmul(A, B, outs=[C])
            func.ReturnOp([])
        return f

    def uniquely_match(self):
        dims = self.dims.values()

        sym_name, input_var, seq_sig = transform.get_seq_signature(
            input_consumed=False,
            has_output=True,
        )

        res_var, global_match_sig = transform.get_match_sig(input_var)
        bb_input_var, bb_header = transform.get_bb_header()

        match_dims = transform.get_match_dims(bb_input_var, dims)

        match_opname = transform.get_match_op_name(bb_input_var, "linalg.matmul")

        tmyield = transform.get_match_structured_terminator(bb_input_var)

        tyield = transform.get_terminator(result=res_var)

        lines = (
            [
                seq_sig,
                "{",
                global_match_sig,
                "{",
                bb_header,
            ]
            + match_dims
            + [match_opname, tmyield, "}", tyield, "}"]
        )

        return sym_name, "\n".join(lines)

    def init_payload(self):
        with InsertionPoint.at_block_begin(self.module.body):
            init_func = func.FuncOp(
                name=self.init_payload_name,
                type=FunctionType.get(inputs=[], results=[self.C_tensor_type]),
                loc=self.loc,
            )
        with InsertionPoint(init_func.add_entry_block()):
            C = self.initialize_tensor((self.i, self.j), 0.0)
            func.ReturnOp([C], loc=self.loc)
        return init_func

    def main(self, frtclock, fprint, fmatmul):
        #
        with InsertionPoint.at_block_begin(self.module.body):
            fmain = func.FuncOp(
                name="main",
                type=FunctionType.get(inputs=[], results=[]),
                loc=self.loc,
            )
        with InsertionPoint(fmain.add_entry_block()):
            #
            A = self.initialize_tensor(
                shape=(self.i, self.k), scalar_value=numpy.random.random()
            )
            B = self.initialize_tensor(
                shape=(self.k, self.j), scalar_value=numpy.random.random()
            )
            #
            callrtclock1 = func.CallOp(frtclock, [], loc=self.loc)
            C = self.initialize_tensor(shape=(self.i, self.j), scalar_value=0.0)
            func.CallOp(fmatmul, [A, B, C], loc=self.loc)
            callrtclock2 = func.CallOp(frtclock, [], loc=self.loc)
            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(fprint, [time], loc=self.loc)
            func.ReturnOp([], loc=self.loc)

        return fmain
