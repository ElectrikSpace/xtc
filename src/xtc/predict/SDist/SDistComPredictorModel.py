#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import re
import shutil
import subprocess
import sys
import tempfile
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

# This lowering pipeline is adapted from the prefix used by
# MlirMppaTarget's MlirProgramToMlirMppaPass._lowering_pipeline(); the two may
# diverge as the SDist cost model evolves independently from MPPA codegen.
_SDIST_PASSES = [
    "sccp",
    "linalg-specialize-generic-ops",
    "sdist-lower-distribution",
    "sdist-insert-kernel-ops",
    "func.func(sdist-fuse-linalg-fill-ops)",
    "sdist-remove-intermediate-subview-ops",
    "convert-sdist-to-sdist-com",
    "sdist-com-group-transfers",
    "sdist-split-for-distributed",
    "sdist-com-apply-double-buffering",
    "sdist-com-tokenize-group-transfers",
    "lower-affine",
]


class SDistComPredictorModel(itf.pred.PredictModel):
    """Estimate schedule cost by lowering to SDist IR and simulating it.

    The MlirProgram and its MlirProgramCompiler only depend on the backend
    (graph, extensions, target, ...), not on the schedule being predicted, so
    they are built once, here, and reused for every call to `predict`. Since
    the underlying MLIR module is mutated in place while lowering a schedule,
    it is reset back to its pristine (unscheduled) state before each use.
    When trace_path is set, each prediction writes a JSON trace there.
    When efficiency_path is set, the simulator applies that YAML overlay.
    """

    def __init__(
        self,
        backend: "itf.pred.Predictor",
        machine_description_path: Path | None,
        trace_path: str | Path | None = None,
        efficiency_path: str | Path | None = None,
    ):
        self._backend = backend

        if machine_description_path is None:
            raise ValueError("Machine description is required")
        self._machine_description_path = machine_description_path.resolve()
        self._trace_path = Path(trace_path).resolve() if trace_path is not None else None
        self._efficiency_path = (
            Path(efficiency_path).resolve() if efficiency_path is not None else None
        )

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
        with tempfile.TemporaryDirectory(prefix="sdist-predict-") as temp_dir:
            ir_path = Path(temp_dir) / "sdist_com.mlir"
            self._run_sdist_pipeline(cast(MlirSchedule, schedule), str(ir_path))
            cmd = [
                *self.cmd_sdist_simulator,
                str(ir_path),
                f"--machine-model={self._machine_description_path}",
                "--double-buffering=false",
            ]
            if self._trace_path is not None:
                cmd.append(f"--trace={self._trace_path}")
            if self._efficiency_path is not None:
                cmd.append(f"--efficiency={self._efficiency_path}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"sdist-simulator failed: {exc.stderr}") from exc
            match = re.search(
                r"^elapsed_cycles:\s*(\d+(?:\.\d+)?)\s*$",
                result.stdout,
                re.MULTILINE,
            )
            if match is None:
                raise ValueError(
                    f"sdist-simulator did not report elapsed_cycles: {result.stdout}"
                )
            return float(match.group(1))

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

    @property
    def cmd_sdist_simulator(self) -> list[str]:
        simulator = shutil.which("sdist-simulator")
        if simulator is not None:
            return [simulator]

        mlir_path = get_mlir_prefix()
        return [f"{mlir_path}/bin/sdist-simulator"]

    @property
    @override
    def backend(self) -> "itf.pred.Predictor":
        return self._backend
