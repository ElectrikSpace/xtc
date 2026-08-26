#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from ..schd.schedule import Schedule
import xtc.itf


class PredictModel(ABC):
    """An abstract implementation of a cost model for a given predictor and schedule.

    A PredictModel takes a backend-specific implementation and schedule and
    estimates its cost. It handles the final stage of converting the optimized
    intermediate representation into a predicted performance metric, without
    having to compile and execute the schedule.
    """

    @abstractmethod
    def predict(self, schedule: Schedule) -> float:
        """Predicts the cost of the implementation for the given schedule.

        Args:
            schedule: The schedule specifying transformations and optimizations

        Returns:
            The predicted cost of the schedule
        """
        ...

    @property
    @abstractmethod
    def backend(self) -> "xtc.itf.pred.Predictor":
        """Returns the predictor associated with this model.

        Returns:
            The predictor this model estimates the cost for
        """
        ...
