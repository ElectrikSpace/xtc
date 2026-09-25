#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import subprocess
from pathlib import Path

import pytest

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
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")

    model = predictor.get_model()

    assert isinstance(model, PredictModel)
    assert isinstance(model, SDistComPredictorModel)
    assert model.backend is predictor


@requires_mlir()
def test_sdist_com_predictor_model_predict_returns_elapsed_cycles(monkeypatch):
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model()
    commands = []

    def simulate(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, "compute_cycles: 10.0\nelapsed_cycles: 42.5\n", ""
        )

    monkeypatch.setattr(subprocess, "run", simulate)

    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})
    scheduler.tile("i", {"i1": 2})
    scheduler.interchange(["i", "j", "k", "i1"])
    schedule = scheduler.schedule()

    cost = model.predict(schedule)

    assert isinstance(cost, float)
    assert cost == 42.5
    assert len(commands) == 1
    assert commands[0][0].endswith("sdist-simulator")
    assert any(arg.startswith("--machine-model=") for arg in commands[0])
    assert not any(arg.startswith("--trace") for arg in commands[0])
    assert not any(arg.startswith("--efficiency") for arg in commands[0])
    assert "--double-buffering=false" in commands[0]


@requires_mlir()
def test_sdist_com_predictor_model_writes_trace_when_requested(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model(trace_path="prediction.json")
    trace_path = tmp_path / "prediction.json"

    def simulate(cmd, **kwargs):
        assert f"--trace={trace_path}" in cmd
        trace_path.write_text('{"traceEvents": []}')
        return subprocess.CompletedProcess(cmd, 0, "elapsed_cycles: 42.5\n", "")

    monkeypatch.setattr(subprocess, "run", simulate)
    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})

    assert model.predict(scheduler.schedule()) == 42.5
    assert trace_path.read_text() == '{"traceEvents": []}'


@requires_mlir()
def test_sdist_com_predictor_model_applies_efficiency_overlay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model(
        trace_path="prediction.json", efficiency_path="cv2_efficiency.yaml"
    )

    def simulate(cmd, **kwargs):
        assert f"--efficiency={tmp_path / 'cv2_efficiency.yaml'}" in cmd
        assert f"--trace={tmp_path / 'prediction.json'}" in cmd
        return subprocess.CompletedProcess(cmd, 0, "elapsed_cycles: 42.5\n", "")

    monkeypatch.setattr(subprocess, "run", simulate)
    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})

    assert model.predict(scheduler.schedule()) == 42.5


@requires_mlir()
def test_sdist_com_predictor_model_rejects_missing_cycles(monkeypatch):
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "no cycles\n", ""),
    )

    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})
    schedule = scheduler.schedule()

    with pytest.raises(ValueError, match="did not report elapsed_cycles"):
        model.predict(schedule)


@requires_mlir()
def test_sdist_com_predictor_model_reports_simulator_failure(monkeypatch):
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model()

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="invalid IR")

    monkeypatch.setattr(subprocess, "run", fail)
    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})

    with pytest.raises(RuntimeError, match="sdist-simulator failed: invalid IR"):
        model.predict(scheduler.schedule())


@requires_mlir()
def test_sdist_com_predictor_model_predict_handles_sequential_schedules(
    capsys, monkeypatch
):
    # The MlirProgram and its MlirProgramCompiler are built once for the
    # model and reused across `predict` calls: make sure the underlying MLIR
    # module is correctly reset between two different schedules, so that
    # lowering the second schedule is not affected by the first one.
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(graph, machine_description_path="machine.yaml")
    model = predictor.get_model()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, "elapsed_cycles: 42.5\n", ""
        ),
    )

    scheduler1 = predictor.get_scheduler()
    scheduler1.define_memory_mesh(axes={"mx": 1})
    scheduler1.define_processor_mesh(axes={"px": 1, "psx": 1})
    scheduler1.tile("i", {"i1": 2})
    scheduler1.interchange(["i", "j", "k", "i1"])
    schedule1 = scheduler1.schedule()

    scheduler2 = predictor.get_scheduler()
    scheduler2.define_memory_mesh(axes={"mx": 1})
    scheduler2.define_processor_mesh(axes={"px": 1, "psx": 1})
    scheduler2.tile("j", {"j1": 4})
    scheduler2.interchange(["j", "i", "k", "j1"])
    schedule2 = scheduler2.schedule()

    cost1 = model.predict(schedule1)
    ir_after_schedule1 = capsys.readouterr().err

    cost2 = model.predict(schedule2)
    ir_after_schedule2 = capsys.readouterr().err

    # Predicting again with the first schedule should reproduce the exact
    # same IR as the first time, showing that the module was properly reset.
    cost1_again = model.predict(schedule1)
    ir_after_schedule1_again = capsys.readouterr().err

    for cost in (cost1, cost2, cost1_again):
        assert isinstance(cost, float)
        assert cost == 42.5

    assert '"./i1"' in ir_after_schedule1
    assert '"./j1"' not in ir_after_schedule1

    assert '"./j1"' in ir_after_schedule2
    assert '"./i1"' not in ir_after_schedule2

    assert ir_after_schedule1 == ir_after_schedule1_again


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
