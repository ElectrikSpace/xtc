#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
import subprocess
import os
from pathlib import Path
from typing import cast
from typing_extensions import override

from mlir.ir import Module
from mlir.passmanager import PassManager

import xtc.itf as itf
from xtc.utils.tools import get_mlir_prefix
from xtc.backends.mlir.MlirCompiler import MlirProgramCompiler
from xtc.backends.mlir.MlirConfig import MlirConfig
from xtc.backends.mlir.MlirProgram import MlirProgram
from xtc.backends.mlir.MlirScheduler import MlirSchedule
from xtc.backends.mlir.MlirTarget import get_target_from_name

from mlir_sdist.trace_analyzer import (
    load_metadata,
    load_machine_model,
    simulate,
)

# The sdist lowering pipeline, up to (and including) the
# `sdist-remove-intermediate-subview-ops` pass. This is a deliberate copy of
# (a prefix of) the pipeline used by MlirMppaTarget's
# MlirProgramToMlirMppaPass._lowering_pipeline(): the two are allowed to
# diverge over time as the SDist cost model evolves independently from the
# actual MPPA code generation pipeline.
_SDIST_PASSES = [
    "sccp",
    "linalg-specialize-generic-ops",
    "sdist-lower-distribution",
    "sdist-insert-kernel-ops",
    "func.func(sdist-fuse-linalg-fill-ops)",
    "sdist-group-transfers",
    "sdist-remove-intermediate-subview-ops",
    "convert-sdist-to-sdist-com",
    "lower-affine",
]


class SDistComPredictorModel(itf.pred.PredictModel):
    """A cost model for the SDist predictor.

    This is a placeholder implementation: it lowers the schedule through the
    sdist pipeline (see `_SDIST_PASSES`), up to (and including) the
    `sdist-remove-intermediate-subview-ops` pass, then prints the resulting
    IR and returns a random value for any given schedule. Actual cost
    estimation is not implemented yet.

    The MlirProgram and its MlirProgramCompiler only depend on the backend
    (graph, extensions, target, ...), not on the schedule being predicted, so
    they are built once, here, and reused for every call to `predict`. Since
    the underlying MLIR module is mutated in place while lowering a schedule,
    it is reset back to its pristine (unscheduled) state before each use.
    """

    def __init__(self, backend: "itf.pred.Predictor", machine_description_path: Path | None):
        self._backend = backend

        # Load machine model
        assert machine_description_path is not None, "Machine description is required"
        # FIXME for debug
        full_path =  os.path.join(machine_description_path)
        self._machine_model = load_machine_model(full_path)

        mlir_backend = backend.mlir_backend
        config = MlirConfig(required_extensions=["sdist"])
        target = get_target_from_name("mppa")(config)

        self._mlir_program = MlirProgram(mlir_backend.xdsl_func, mlir_backend.no_alias)
        # Pristine (unscheduled) IR, used to reset the module before lowering
        # each new schedule.
        self._pristine_module_source = str(self._mlir_program.mlir_module)

        self._compiler = MlirProgramCompiler(
            mlir_program=self._mlir_program,
            mlir_schedule=None,
            concluding_passes=mlir_backend.concluding_passes,
            always_vectorize=mlir_backend.always_vectorize,
            config=config,
            target=target,
        )

    @override
    def predict(self, schedule: "itf.schd.Schedule") -> float:
        sdist_com_dump_file = "/tmp/sdist_com.mlir"
        loopnest_dump_file = "/tmp/loopnest.json"
        log_file = "/tmp/trace.log"

        # Lower down to SDistCom Dialect
        self._run_sdist_pipeline(cast(MlirSchedule, schedule), sdist_com_dump_file)
        # Extract the decorated LoopNest
        self._run_loopnest_extraction(sdist_com_dump_file, loopnest_dump_file)
        # Load the LoopNest in the simulator
        loopnest = load_metadata(loopnest_dump_file)

        # Run the simulator
        #result = simulate(loopnest, self._machine_model, log_file, double_buffering=True)
        result = simulate(loopnest, self._machine_model, log_file, double_buffering=False)
        return result.total_cycles

    def _reset_mlir_program(self) -> None:
        self._mlir_program.module = Module.parse(
            self._pristine_module_source, context=self._mlir_program.mlir_context
        )

    def _run_sdist_pipeline(self, schedule: MlirSchedule, sdist_com_dump_file: str) -> None:
        self._reset_mlir_program()
        self._compiler._mlir_schedule = schedule

        self._compiler.mlir_insert_transform_pass()
        self._compiler.mlir_apply_transform_pass()
        self._compiler.mlir_apply_tensor_lowering_pass()

        self._run_sdist_lowering(self._mlir_program)

        print(
            f"// -----// IR Dump After {_SDIST_PASSES[-1]} //----- //", file=sys.stderr
        )
        print(str(self._mlir_program.mlir_module), file=sys.stderr)
        with open(sdist_com_dump_file, "w") as outf:
            outf.write(str(self._mlir_program.mlir_module))

    def _run_sdist_lowering(self, mlir_program: MlirProgram) -> None:
        # Run the (local copy of the) sdist lowering pipeline defined above.
        assert "sdist" in mlir_program.mlir_extensions
        new_passes = []
        for p in _SDIST_PASSES:
            new_passes.append(p)
            new_passes.append("canonicalize")
            new_passes.append("cse")

        mlir_program.mlir_context.allow_unregistered_dialects = True
        pm = PassManager(context=mlir_program.mlir_context)
        pm.enable_verifier(False)
        for opt in new_passes:
            pm.add(opt)  # type: ignore # no attribute add?
        pm.run(mlir_program.mlir_module.operation)
        mlir_program.mlir_context.allow_unregistered_dialects = False

    def _execute_command(
        self,
        cmd: list[str],
        input_pipe: str | None = None,
        pipe_stdoutput: bool = True,
    ) -> subprocess.CompletedProcess:
        pretty_cmd = "| " if input_pipe else ""
        pretty_cmd += " ".join(cmd)
        #if self._config.debug:
        #    print(f"> exec: {pretty_cmd}", file=sys.stderr)

        if input_pipe and pipe_stdoutput:
            result = subprocess.run(
                cmd, input=input_pipe, stdout=subprocess.PIPE, text=True
            )
        elif input_pipe and not pipe_stdoutput:
            result = subprocess.run(cmd, input=input_pipe, text=True)
        elif not input_pipe and pipe_stdoutput:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        else:
            result = subprocess.run(cmd, text=True)
        return result

    def _run_loopnest_extraction(self, sdist_com_dump_file: str, loopnest_dump_file: str) -> None:
        cmd = self.cmd_sdist_extract + [
            "--sdist-get-infos",
            sdist_com_dump_file,
            "-o",
            loopnest_dump_file,
        ]
        exe_process = self._execute_command(cmd=cmd)
        assert exe_process.returncode == 0

    @property
    def cmd_sdist_extract(self):
        mlir_path = get_mlir_prefix()
        return [f"{mlir_path}/bin/sdist-extract"]

    @property
    @override
    def backend(self) -> "itf.pred.Predictor":
        return self._backend
