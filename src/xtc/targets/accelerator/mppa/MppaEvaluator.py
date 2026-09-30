#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import ctypes
from typing import Any
from typing_extensions import override
import numpy as np

import xtc.targets.accelerator.mppa as mppa
import xtc.itf as itf
from xtc.errors import XtcRuntimeError
from xtc.runtimes.accelerator.mppa import MppaDevice
from xtc.runtimes.types.ndarray import NDArray
from xtc.utils.isolation import mark_isolated_stage, run_isolated
from xtc.utils.evaluation import (
    ensure_ndarray_parameters,
    validate_outputs,
    evaluate_performance,
    copy_outputs,
)

__all__ = [
    "MppaEvaluator",
    "MppaExecutor",
]


class MppaEvaluator(itf.exec.Evaluator):
    def __init__(self, module: "mppa.MppaModule", **kwargs: Any) -> None:
        self._module = module
        self._isolate_execution = kwargs.get("isolate_execution", True)
        self._repeat = kwargs.get("repeat", 2)
        self._min_repeat_ms = kwargs.get("min_repeat_ms", 0)
        assert self._min_repeat_ms == 0, "min_repeat_ms > 0 is not supported yet"
        self._number = kwargs.get("number", 1)
        assert self._number == 1, "number > 1 is not supported yet"  # TODO
        # TODO support min_repeat_ms and number
        # But execution on MPPA has almost no noise except DDR refresh
        self._validate = kwargs.get("validate", False)
        self._parameters = kwargs.get("parameters")
        self._init_zero = kwargs.get("init_zero", False)
        self._np_inputs_spec = kwargs.get(
            "np_inputs_spec", self._module._np_inputs_spec
        )
        self._np_outputs_spec = kwargs.get(
            "np_outputs_spec", self._module._np_outputs_spec
        )
        self._reference_impl = kwargs.get(
            "reference_impl", self._module._reference_impl
        )
        self._pmu_counters = kwargs.get("pmu_counters", [])

        assert self._module.file_type == "shlib", "only support shlib for evaluation"

    @override
    def evaluate(self) -> tuple[list[float], int, str]:
        assert self._module._bare_ptr, "bare_ptr is not supported for evaluation"
        if not self._isolate_execution:
            return self._evaluate_native()[0]

        existing = MppaDevice._instance
        if existing is not None and (
            existing.mppa_initialized
            or existing.lib_loader is not None
            or existing.loaded_kernels
        ):
            raise XtcRuntimeError(
                "MPPA native state is already initialized in the parent process; "
                "create and use the evaluator before initializing the device, "
                "or use isolate_execution=False",
                stage="MPPA device initialization",
            )

        snapshots: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        layouts: tuple[list[list[int] | None], list[list[int] | None]] | None = None
        if self._parameters is not None:
            snapshots = ([], [])
            layouts = ([], [])
            for source, arrays, array_layouts in zip(
                self._parameters, snapshots, layouts
            ):
                for item in source:
                    if isinstance(item, NDArray) and item.is_on_device():
                        raise XtcRuntimeError(
                            "Device-backed NDArray parameters cannot be inherited "
                            "by an isolated MPPA worker; use host arrays",
                            stage="MPPA parameter preparation",
                        )
                    arrays.append(
                        item.numpy().copy()
                        if isinstance(item, NDArray)
                        else np.array(item, copy=True)
                    )
                    array_layouts.append(
                        item.layout if isinstance(item, NDArray) else None
                    )

        def operation() -> tuple[tuple[list[float], int, str], list[np.ndarray] | None]:
            mark_isolated_stage("MPPA device initialization")
            worker_parameters: tuple[list[NDArray], list[NDArray]] | None = None
            if snapshots is not None and layouts is not None:
                worker_parameters = (
                    [
                        NDArray(array, layout=layout)
                        for array, layout in zip(snapshots[0], layouts[0])
                    ],
                    [
                        NDArray(array, layout=layout)
                        for array, layout in zip(snapshots[1], layouts[1])
                    ],
                )
            return self._evaluate_native(worker_parameters, isolated=True)

        result, outputs = run_isolated(
            operation, error_type=XtcRuntimeError, stage="MPPA device initialization"
        )
        if outputs is not None and self._parameters is not None:
            for target, output in zip(self._parameters[1], outputs):
                if isinstance(target, NDArray):
                    if target.is_transposed():
                        output = np.ascontiguousarray(output.T)
                    assert target.handle is not None
                    target._copy_from(
                        target.handle, output.ctypes.data_as(ctypes.c_void_p)
                    )
                else:
                    np.copyto(target, output)
        return result

    def _evaluate_native(
        self,
        parameters_override: tuple[list[NDArray], list[NDArray]] | None = None,
        *,
        isolated: bool = False,
    ) -> tuple[tuple[list[float], int, str], list[np.ndarray] | None]:
        device = MppaDevice(self._module._mppa_config)
        mark_isolated_stage("MPPA device initialization")
        device.init_device()
        mark_isolated_stage("MPPA module loading")
        device.load_module(self._module)
        sym = self._module.payload_name
        func = device.get_module_function(self._module, sym)
        results: tuple[list[float], int, str] = ([], 0, "")
        validation_failed = False

        try:
            mark_isolated_stage("MPPA parameter preparation")
            parameters = parameters_override or ensure_ndarray_parameters(
                self._parameters,
                self._np_inputs_spec,
                self._np_outputs_spec,
                self._init_zero,
            )

            if self._validate:
                mark_isolated_stage("MPPA output validation")
                results = validate_outputs(func, parameters, self._reference_impl)
                validation_failed = results[1] != 0

            if not validation_failed:
                mark_isolated_stage("MPPA performance evaluation")
                results = evaluate_performance(
                    func,
                    parameters,
                    self._pmu_counters,
                    self._repeat,
                    self._number,
                    self._min_repeat_ms,
                    device,
                )

            if isolated and self._parameters is not None:
                mark_isolated_stage("MPPA output copyback")
                return results, [output.numpy() for output in parameters[1]]
            if self._parameters is not None:
                copy_outputs(parameters, self._parameters)
            return results, None
        finally:
            mark_isolated_stage("MPPA module unloading")
            device.unload_module(self._module)

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._module


class MppaExecutor(itf.exec.Executor):
    def __init__(self, module: "mppa.MppaModule", **kwargs: Any) -> None:
        self._evaluator = MppaEvaluator(
            module=module,
            repeat=1,
            min_repeat_ms=0,
            number=1,
            **kwargs,
        )

    @override
    def execute(self) -> int:
        results, code, err_msg = self._evaluator.evaluate()
        return code

    @property
    @override
    def module(self) -> itf.comp.Module:
        return self._evaluator.module
