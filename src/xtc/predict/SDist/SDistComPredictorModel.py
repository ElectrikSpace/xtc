#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import random
import sys
from typing import cast
from typing_extensions import override

from mlir.ir import Module
from mlir.passmanager import PassManager

import xtc.itf as itf
from xtc.backends.mlir.MlirCompiler import MlirProgramCompiler
from xtc.backends.mlir.MlirConfig import MlirConfig
from xtc.backends.mlir.MlirProgram import MlirProgram
from xtc.backends.mlir.MlirScheduler import MlirSchedule
from xtc.backends.mlir.MlirTarget import get_target_from_name

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

    def __init__(self, backend: "itf.pred.Predictor"):
        self._backend = backend

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
        self._run_sdist_pipeline(cast(MlirSchedule, schedule))
        return random.random()

    def _reset_mlir_program(self) -> None:
        self._mlir_program.module = Module.parse(
            self._pristine_module_source, context=self._mlir_program.mlir_context
        )

    def _run_sdist_pipeline(self, schedule: MlirSchedule) -> None:
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
    @override
    def backend(self) -> "itf.pred.Predictor":
        return self._backend
