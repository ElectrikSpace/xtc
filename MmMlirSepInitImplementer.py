#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess
import numpy

from mlir.ir import *
import mlir
from mlir.dialects import arith, builtin, func, linalg, tensor

from MmMlirImplementer import MmMlirImplementer
import transform


class MmMlirSepInitImplementer(MmMlirImplementer):
    def payload(self):
        with InsertionPoint.at_block_begin(self.module.body), self.loc as loc:
            f = func.FuncOp(
                name=self.payload_name,
                type=FunctionType.get(
                    inputs=[self.A_tensor_type, self.B_tensor_type, self.C_tensor_type],
                    results=[self.C_tensor_type],
                ),
            )
            entry_block = f.add_entry_block()
        with InsertionPoint(entry_block), self.loc as loc:
            A = f.entry_block.arguments[0]
            B = f.entry_block.arguments[1]
            C_init = f.entry_block.arguments[2]
            matmul = linalg.matmul(A, B, outs=[C_init])
            func.ReturnOp([matmul])
        return f

    def main(self, frtclock, fprint, fmatmul, init_payload):
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
                shape=(self.i, self.k), value=numpy.random.random()
            )
            #
            B = self.initialize_tensor(
                shape=(self.k, self.j), value=numpy.random.random()
            )
            #
            callrtclock1 = func.CallOp(frtclock, [], loc=self.loc)
            C_init = func.CallOp(init_payload, [], loc=self.loc)
            C = func.CallOp(fmatmul, [A, B, C_init], loc=self.loc)
            callrtclock2 = func.CallOp(frtclock, [], loc=self.loc)
            time = arith.SubFOp(callrtclock2, callrtclock1, loc=self.loc)
            func.CallOp(fprint, [time], loc=self.loc)
            func.ReturnOp([], loc=self.loc)

        return fmain
