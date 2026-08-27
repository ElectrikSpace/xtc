#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path
from typing import Any
from typing_extensions import override

import xtc.itf as itf
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend

from .SDistComPredictorModel import SDistComPredictorModel


class SDistPredictor(itf.pred.Predictor):
    """A Predictor implementation for the MLIR + SDist backend.

    It builds its own MlirGraphBackend from the input Graph, reusing the
    Graph and Scheduler (and thus Schedule) implementations provided by that
    backend, and exposes a dedicated SDistComPredictorModel to estimate the
    cost of a schedule.

    The machine targeted by the SDist predictor is described by a YAML file
    whose path is kept around as-is: parsing and interpreting its content is
    delegated to a dependency project.
    """

    def __init__(
        self,
        graph: "itf.graph.Graph",
        machine_description_path: str | Path | None = None,
        **backend_kwargs: Any,
    ):
        self._backend = MlirGraphBackend(graph, **backend_kwargs)
        self._machine_description_path = (
            Path(machine_description_path)
            if machine_description_path is not None
            else None
        )

    @override
    def get_scheduler(self, **kwargs: Any) -> "itf.schd.Scheduler":
        return self._backend.get_scheduler(**kwargs)

    @override
    def get_model(self, **kwargs: Any) -> SDistComPredictorModel:
        return SDistComPredictorModel(self)

    @property
    @override
    def graph(self) -> "itf.graph.Graph":
        return self._backend.graph

    @property
    def mlir_backend(self) -> MlirGraphBackend:
        """Returns the underlying MlirGraphBackend created by this predictor."""
        return self._backend

    @property
    def machine_description_path(self) -> Path | None:
        """Returns the path to the YAML machine description file, if any."""
        return self._machine_description_path
