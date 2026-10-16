#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import importlib
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlir_utils import requires_mlir, matmul_graph, matmul_impl

from xtc.predict.SDist.SDistPredictor import SDistPredictor
from xtc.predict.SDist.SDistComPredictorModel import SDistComPredictorModel
from xtc.itf.pred.predictor import Predictor
from xtc.itf.pred.model import PredictModel
from xtc.itf.schd import Schedule

MACHINES_DIR = Path(__file__).parents[2] / "machines"


def _buffered_matmul_schedule(predictor: SDistPredictor) -> Schedule:
    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 4, "psy": 4})
    scheduler.tile("i", {"i1": 64})
    scheduler.tile("j", {"j1": 64})
    scheduler.tile("k", {"k1": 64})
    scheduler.tile("i", {"iv": 8})
    scheduler.tile("j", {"jv": 8})
    scheduler.interchange(["j", "i", "k", "i1", "j1", "iv", "jv", "k1"])
    scheduler.pack_at("i1", 0)
    scheduler.pack_at("i1", 1)
    scheduler.buffer_at("k")
    scheduler.distribute("i1", "psx")
    scheduler.distribute("j1", "psy")
    scheduler.vectorize(["iv", "jv", "k1"])
    return scheduler.schedule()


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
    # Buffering is encoded in the lowered IR, not a simulator CLI option.
    assert not any(arg.startswith("--double-buffering") for arg in commands[0])


@requires_mlir()
def test_sdist_com_predictor_model_selects_double_buffering_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = SDistPredictor(
        matmul_graph(128, 128, 128, "float32", "matmul"),
        machine_description_path="machine.yaml",
    )
    model = predictor.get_model()
    schedule = _buffered_matmul_schedule(predictor)
    model_module = importlib.import_module("xtc.predict.SDist.SDistComPredictorModel")
    pass_manager = model_module.PassManager
    managers: list[Mock] = []
    lowered_ir: list[str] = []

    def record_passes(**kwargs: object) -> Mock:
        manager = Mock(wraps=pass_manager(**kwargs))
        managers.append(manager)
        return manager

    def simulate(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        lowered_ir.append(Path(cmd[1]).read_text())
        return subprocess.CompletedProcess(cmd, 0, "elapsed_cycles: 42.5\n", "")

    monkeypatch.setattr(model_module, "PassManager", record_passes)
    monkeypatch.setattr(subprocess, "run", simulate)

    assert model.predict(schedule, double_buffering=False) == 42.5
    assert model.predict(schedule, double_buffering=True) == 42.5
    assert model.predict(schedule) == 42.5
    assert model.predict(schedule, double_buffering=False) == 42.5

    for manager, enabled in zip(managers, [False, True, True, False]):
        passes = [call.args[0] for call in manager.add.call_args_list]
        assert ("sdist-split-for-distributed" in passes) is enabled
        assert ("sdist-com-apply-double-buffering" in passes) is enabled
        assert "sdist-com-tokenize-group-transfers" in passes
    assert lowered_ir[0] == lowered_ir[3]
    assert lowered_ir[1] == lowered_ir[2]
    assert lowered_ir[0] != lowered_ir[1]


@requires_mlir()
@pytest.mark.parametrize("double_buffering", [0, 1, None, "true"])
def test_sdist_com_predictor_model_rejects_non_boolean_double_buffering(
    double_buffering: int | str | None,
) -> None:
    predictor = SDistPredictor(
        matmul_graph(4, 32, 512, "float32", "matmul"),
        machine_description_path="machine.yaml",
    )
    model = predictor.get_model()

    with pytest.raises(TypeError, match="double_buffering must be a bool"):
        model.predict(
            predictor.get_scheduler().schedule(), double_buffering=double_buffering
        )


@requires_mlir()
def test_sdist_com_predictor_model_simulates_double_buffering() -> None:
    predictor = SDistPredictor(
        matmul_graph(128, 128, 128, "float32", "matmul"),
        machine_description_path=MACHINES_DIR / "cv2.yaml",
    )
    model = predictor.get_model()
    if not Path(model.cmd_sdist_simulator[0]).is_file():
        pytest.skip("requires sdist-simulator")
    schedule = _buffered_matmul_schedule(predictor)

    single_cycles = model.predict(schedule, double_buffering=False)
    double_cycles = model.predict(schedule, double_buffering=True)
    assert 0 < double_cycles < single_cycles
    assert model.predict(schedule) == double_cycles
    assert model.predict(schedule, double_buffering=False) == single_cycles


@requires_mlir()
def test_sdist_com_predictor_model_prints_stdout_when_verbose(capsys, monkeypatch):
    graph = matmul_graph(4, 32, 512, "float32", "matmul")
    predictor = SDistPredictor(
        graph, machine_description_path="machine.yaml", verbose=True
    )
    model = predictor.get_model()
    simulator_stdout = "compute_cycles: 10.0\nelapsed_cycles: 42.5\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, simulator_stdout, ""),
    )

    scheduler = predictor.get_scheduler()
    scheduler.define_memory_mesh(axes={"mx": 1})
    scheduler.define_processor_mesh(axes={"px": 1, "psx": 1})

    assert model.predict(scheduler.schedule()) == 42.5
    assert capsys.readouterr().out == simulator_stdout


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

    predictor = SDistPredictor(graph, machine_description_path=machine_description_path)

    assert predictor.machine_description_path == machine_description_path
