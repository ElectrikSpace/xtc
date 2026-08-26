#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from pathlib import Path

from mlir_utils import requires_mlir, matmul_graph, matmul_impl

from xtc.predict.SDist.SDistPredictor import SDistPredictor
from xtc.predict.SDist.SDistComPredictorModel import SDistComPredictorModel
from xtc.itf.pred.predictor import Predictor
from xtc.itf.pred.model import PredictModel

MACHINES_DIR = Path(__file__).parents[2] / "machines"


@requires_mlir()
def test_sdist_predictor_is_a_predictor():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)

    assert isinstance(predictor, Predictor)
    assert predictor.graph is graph


@requires_mlir()
def test_sdist_predictor_reuses_backend_scheduler():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)
    backend = matmul_impl(4, 32, 512, "float32", "matmul")

    scheduler = predictor.get_scheduler()

    # The scheduler produced by the predictor should behave exactly like the
    # one produced directly by the underlying MLIR backend, since the graph
    # and schedule implementations are reused as-is.
    assert type(scheduler) is type(backend.get_scheduler())


@requires_mlir()
def test_sdist_predictor_get_model_returns_com_predictor_model():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)

    model = predictor.get_model()

    assert isinstance(model, PredictModel)
    assert isinstance(model, SDistComPredictorModel)
    assert model.backend is predictor


@requires_mlir()
def test_sdist_com_predictor_model_predict_returns_random_value_in_unit_range():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)
    model = predictor.get_model()

    scheduler = predictor.get_scheduler()
    scheduler.tile("i", {"i1": 2})
    scheduler.interchange(["i", "j", "k", "i1"])
    schedule = scheduler.schedule()

    cost = model.predict(schedule)

    assert isinstance(cost, float)
    assert 0.0 <= cost <= 1.0


@requires_mlir()
def test_sdist_com_predictor_model_predict_is_random():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)
    model = predictor.get_model()

    scheduler = predictor.get_scheduler()
    schedule = scheduler.schedule()

    costs = {model.predict(schedule) for _ in range(20)}

    # It is astronomically unlikely for 20 random floats to collide.
    assert len(costs) > 1


@requires_mlir()
def test_sdist_predictor_defaults_to_no_machine_description():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph)

    assert predictor.machine_description_path is None


@requires_mlir()
def test_sdist_predictor_accepts_machine_description_path_kwarg():
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    machine_description_path = MACHINES_DIR / "cv2.yaml"

    predictor = SDistPredictor(
        graph, machine_description_path=machine_description_path
    )

    assert predictor.machine_description_path == machine_description_path
