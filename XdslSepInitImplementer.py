#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC
import subprocess

from xdsl.parser import Parser
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.ir import Block, Region, MLContext, Operation
from xdsl.dialects.builtin import (
    ModuleOp,
    DenseIntOrFPElementsAttr,
    TensorType,
    MemRefType,
    f32,
    f64,
)
from xdsl.dialects import func, arith, linalg
from AbsImplementer import AbsImplementer
from XdslImplementer import XdslImplementer
import transform


class XdslSepInitImplementer(XdslImplementer):
    def payload(self):
        # Fetch data
        operands = self.source_op.operands
        inputs = self.source_op.inputs
        inputs_types = [o.type for o in operands]
        results_types = [r.type for r in self.source_op.results]
        #
        payload = Block(arg_types=inputs_types)
        concrete_operands = list(payload.args)
        value_mapper = {o: p for o, p in zip(operands, concrete_operands)}

        new_op = self.source_op.clone(value_mapper=value_mapper)
        payload.add_ops([new_op, func.Return(new_op)])
        payload_func = func.FuncOp.from_region(
            self.payload_name, inputs_types, results_types, Region(payload)
        )
        return payload_func

    def main(self, ext_rtclock, ext_printF64, payload_func, init_payload):
        results_types = [r.type for r in self.source_op.results]
        #
        inputs = self.inputs_init()
        rtclock_call1 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        outputs = self.outputs_init()
        init_payload_call = func.Call(init_payload.sym_name.data, [], results_types)
        payload_call = func.Call(
            payload_func.sym_name.data,
            inputs + init_payload_call.results,
            results_types,
        )
        rtclock_call2 = func.Call(ext_rtclock.sym_name.data, [], [f64])
        elapsed = arith.Subf(rtclock_call2, rtclock_call1)
        print_elapsed = func.Call(ext_printF64.sym_name.data, [elapsed], [])
        main = Block()
        main.add_ops(
            inputs
            + [
                rtclock_call1,
                init_payload_call,
                payload_call,
                rtclock_call2,
                elapsed,
                print_elapsed,
                func.Return(),
            ]
        )
        main_func = func.FuncOp.from_region("main", [], [], Region(main))
        return main_func
