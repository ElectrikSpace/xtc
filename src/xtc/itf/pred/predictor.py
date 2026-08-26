#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any

from ..graph.graph import Graph
from ..schd.scheduler import Scheduler
from .model import PredictModel


class Predictor(ABC):
    """An abstract implementation of specific Graph implementation used for prediction.

    A Predictor is constructed from an input Graph and provides backend-specific
    implementations of the graph operations, similarly to a Backend. It serves as a
    bridge between the abstract graph representation and concrete cost-model
    implementations used to estimate the performance of a Schedule without actually
    compiling and executing it.

    The Predictor provides access to an associated Scheduler for applying
    transformations, and to a PredictModel for estimating the cost of the
    resulting schedules.
    """

    @abstractmethod
    def get_scheduler(self, **kwargs: Any) -> Scheduler:
        """Returns the scheduler associated with this implementation.

        Args:
            kwargs: scheduler configuration

        Returns:
            The scheduler for applying transformations
        """
        ...

    @abstractmethod
    def get_model(self, **kwargs: Any) -> PredictModel:
        """Returns the prediction model associated with this implementation.

        Args:
            kwargs: model configuration

        Returns:
            The model for predicting the cost of a schedule
        """
        ...

    @property
    @abstractmethod
    def graph(self) -> Graph:
        """Returns the graph being implemented.

        Returns:
            The source graph for this implementation
        """
        ...
