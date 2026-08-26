#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import pytest

from xtc.itf.pred.predictor import Predictor
from xtc.itf.pred.model import PredictModel


def test_predictor_is_abstract():
    with pytest.raises(TypeError):
        Predictor()  # type: ignore[abstract]


def test_predict_model_is_abstract():
    with pytest.raises(TypeError):
        PredictModel()  # type: ignore[abstract]


class DummyPredictModel(PredictModel):
    def __init__(self, predictor: "DummyPredictor", cost: float):
        self._predictor = predictor
        self._cost = cost

    def predict(self, schedule) -> float:
        return self._cost

    @property
    def backend(self) -> "DummyPredictor":
        return self._predictor


class DummyScheduler:
    pass


class DummyPredictor(Predictor):
    def __init__(self, graph, cost: float = 42.0):
        self._graph = graph
        self._cost = cost

    def get_scheduler(self, **kwargs) -> DummyScheduler:
        return DummyScheduler()

    def get_model(self, **kwargs) -> DummyPredictModel:
        return DummyPredictModel(self, self._cost)

    @property
    def graph(self):
        return self._graph


def test_predictor_concrete_subclass():
    graph = object()
    predictor = DummyPredictor(graph)

    assert isinstance(predictor, Predictor)
    assert predictor.graph is graph
    assert isinstance(predictor.get_scheduler(), DummyScheduler)

    model = predictor.get_model()
    assert isinstance(model, PredictModel)
    assert model.backend is predictor


def test_predict_model_predict_returns_cost():
    predictor = DummyPredictor(object(), cost=13.0)
    model = predictor.get_model()

    cost = model.predict(schedule=None)

    assert cost == 13.0


def test_predictor_missing_abstract_method_cannot_be_instantiated():
    class IncompletePredictor(Predictor):
        def get_scheduler(self, **kwargs):
            return DummyScheduler()

        # get_model and graph are intentionally not implemented

    with pytest.raises(TypeError):
        IncompletePredictor()  # type: ignore[abstract]


def test_predict_model_missing_abstract_method_cannot_be_instantiated():
    class IncompletePredictModel(PredictModel):
        def predict(self, schedule) -> float:
            return 0.0

        # backend is intentionally not implemented

    with pytest.raises(TypeError):
        IncompletePredictModel()  # type: ignore[abstract]
