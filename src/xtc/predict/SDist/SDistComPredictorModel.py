#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import random
from typing_extensions import override

import xtc.itf as itf


class SDistComPredictorModel(itf.pred.PredictModel):
    """A cost model for the SDist predictor.

    This is a placeholder implementation: it does not perform any actual
    cost estimation and simply returns a random value for any given
    schedule.
    """

    def __init__(self, backend: "itf.pred.Predictor"):
        self._backend = backend

    @override
    def predict(self, schedule: "itf.schd.Schedule") -> float:
        return random.random()

    @property
    @override
    def backend(self) -> "itf.pred.Predictor":
        return self._backend
