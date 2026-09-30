#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import traceback

import pytest

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend
from xtc.schedules import exceptions
from xtc.schedules.exceptions import ScheduleValidationError


def matmul_scheduler(i=512, j=512, k=512):
    a = O.tensor((i, k), "float32", name="A")
    b = O.tensor((k, j), "float32", name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")
    impl = Backend(gb.graph)
    return impl, impl.get_scheduler()


def test_invalid_py_tile_order_is_rejected():
    _, sch = matmul_scheduler()
    sch.tile("i", {"i0": 512, "i1": 256})
    sch.tile("j", {"j0": 128, "j1": 256})
    sch.interchange(["i", "j", "j0", "i0", "i1", "j1", "k"])

    with pytest.raises(
        ScheduleValidationError,
        match=r"Inner tile \./j1 \(256\) exceeds outer tile \./j0 \(128\)",
    ):
        sch.schedule()


@pytest.mark.parametrize("inner", [64, 128])
def test_valid_tile_order_and_default_interchange(inner):
    _, sch = matmul_scheduler()
    sch.tile("j", {"j0": 128, "j1": inner})
    assert sch.schedule().schedule_impl[-1].permutation["."] == [
        "./i",
        "./j",
        "./k",
        "./j0",
        "./j1",
    ]


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_tile_size_is_rejected(size):
    _, sch = matmul_scheduler()
    sch.tile("j", {"j0": size})
    with pytest.raises(ScheduleValidationError, match=r"Tile \./j0.*positive size"):
        sch.schedule()


def test_invalid2_py_tiles_exceed_problem_dimensions():
    _, sch = matmul_scheduler(128, 128, 128)
    sch.tile("i", {"i0": 512, "i1": 512, "i2": 16})
    sch.tile("j", {"j0": 1024, "j1": 128, "j2": 32})
    sch.interchange(["j", "i", "j0", "i0", "i1", "j1", "i2", "j2", "k"])

    with pytest.raises(
        ScheduleValidationError,
        match=r"Tile \./i0 \(512\) exceeds problem dimension i \(128\)",
    ) as error:
        sch.schedule()
    assert "  Dimension: i" in str(error.value)

    sch.tile("i", {"i0": 128, "i1": 128})
    with pytest.raises(
        ScheduleValidationError,
        match=r"Tile \./j0 \(1024\) exceeds problem dimension j \(128\)",
    ):
        sch.schedule()


def test_tile_equal_to_extent_is_valid_after_renaming_dims():
    _, sch = matmul_scheduler(64, 128, 256)
    sch.set_dims(["I", "J", "K"])
    sch.tile("I", {"I0": 64})
    sch.tile("J", {"J0": 128})
    sch.tile("K", {"K0": 256})
    assert sch.schedule().schedule_impl[-1].dim_sizes == {
        "I": 64,
        "J": 128,
        "K": 256,
    }

    sch.tile("K", {"K0": 257})
    with pytest.raises(
        ScheduleValidationError,
        match=r"Tile \./K0 \(257\) exceeds problem dimension K \(256\)",
    ):
        sch.schedule()


def test_split_root_tile_cannot_exceed_full_dimension():
    _, sch = matmul_scheduler(128, 128, 128)
    sch.split("i", {"i_lo": 0, "i_hi": 64})
    sch.tile("j", {"j0": 129}, root="./i_lo")
    sch.interchange(["k", "i_lo", "i_hi"])
    sch.interchange(["j", "j0"], root="./i_lo")
    sch.interchange(["j"], root="./i_hi")
    with pytest.raises(
        ScheduleValidationError,
        match=r"Tile \./i_lo/j0 \(129\) exceeds problem dimension j \(128\)",
    ):
        sch.schedule()


@pytest.mark.parametrize(
    ("permutation", "message"),
    [
        (["i", "j", "k"], r"missing axes \['\./j0'\]"),
        (["i", "j", "k", "j0", "j1"], r"unknown axes \['\./j1'\]"),
        (["i", "j", "k", "j0", "j0"], r"Duplicate interchange axes"),
        (["i", "j", "j0"], r"missing base dimensions \['k'\]"),
    ],
)
def test_interchange_must_contain_exactly_existing_axes(permutation, message):
    _, sch = matmul_scheduler()
    sch.tile("j", {"j0": 128})
    sch.interchange(permutation)
    with pytest.raises(ScheduleValidationError, match=message):
        sch.schedule()


def test_interchange_must_use_tile_from_its_own_root():
    _, sch = matmul_scheduler()
    sch.split("i", {"i_lo": 0, "i_hi": 256})
    sch.tile("j", {"j0": 128}, root="./i_lo")
    sch.tile("j", {"j0": 128}, root="./i_hi")
    sch.interchange(["k", "i_lo", "i_hi"])
    sch.interchange(["j", "j0"], root="./i_lo")
    sch.interchange(["j", "j0"], root="./i_hi")
    assert sch.schedule()

    sch.interchange(["j", "j0", "j1"], root="./i_hi")
    with pytest.raises(
        ScheduleValidationError, match=r"unknown axes \['\./i_hi/j1'\]"
    ):
        sch.schedule()


def test_split_child_cannot_omit_its_tile():
    _, sch = matmul_scheduler()
    sch.split("i", {"i_lo": 0, "i_hi": 256})
    sch.tile("j", {"j0": 128}, root="./i_lo")
    sch.interchange(["k", "i_lo", "i_hi"])
    sch.interchange(["j"], root="./i_lo")
    sch.interchange(["j"], root="./i_hi")
    with pytest.raises(
        ScheduleValidationError, match=r"Interchange missing axes"
    ):
        sch.schedule()


def test_split_roots_use_their_own_qualified_axes():
    _, sch = matmul_scheduler()
    sch.split("i", {"i_lo": 0, "i_hi": 256}, root="j")
    sch.tile("k", {"k0": 128, "k1": 64}, root="j/i_lo")
    sch.interchange(["j", "i_lo", "i_hi"], root="j")
    sch.interchange(["k", "k0", "k1"], root="j/i_lo")
    sch.interchange(["k"], root="j/i_hi")
    assert sch.schedule()

    sch.tile("k", {"k1": 256}, root="j/i_lo")
    with pytest.raises(
        ScheduleValidationError,
        match=r"Inner tile j/i_lo/k1 \(256\) exceeds outer tile j/i_lo/k0 \(128\)",
    ):
        sch.schedule()


def test_split_root_must_include_its_split_axes():
    _, sch = matmul_scheduler()
    sch.split("i", {"i_lo": 0, "i_hi": 256})
    sch.interchange(["j", "k", "i_lo"])
    sch.interchange(["j", "k"], root="./i_lo")
    sch.interchange(["j", "k"], root="./i_hi")
    with pytest.raises(
        ScheduleValidationError, match=r"Interchange missing axes \['\./i_hi'\]"
    ):
        sch.schedule()


def test_compile_revalidates_before_generating_ir(monkeypatch):
    impl, sch = matmul_scheduler()
    sch.tile("j", {"j0": 128, "j1": 64})
    sched = sch.schedule()
    sched.schedule_impl[-1].tiles["j"]["./j1"] = 256

    def fail_if_called():
        pytest.fail("Invalid schedule reached IR generation")

    compiler = impl.get_compiler()
    monkeypatch.setattr(compiler, "generate_program", fail_if_called)
    with pytest.raises(ScheduleValidationError, match="Inner tile"):
        compiler.compile(sched)


def test_compile_revalidates_problem_extents_before_generating_ir(monkeypatch):
    impl, sch = matmul_scheduler(128, 128, 128)
    sch.tile("i", {"i0": 128})
    sched = sch.schedule()
    sched.schedule_impl[-1].tiles["i"]["./i0"] = 512

    compiler = impl.get_compiler()
    monkeypatch.setattr(
        compiler,
        "generate_program",
        lambda: pytest.fail("Invalid schedule reached IR generation"),
    )
    with pytest.raises(
        ScheduleValidationError,
        match=r"Tile \./i0 \(512\) exceeds problem dimension i \(128\)",
    ):
        compiler.compile(sched)


def test_every_node_is_checked():
    _, sch = matmul_scheduler()
    sched = sch.schedule()
    node = sched.schedule_impl[-1]
    sched.schedule_impl.append(
        replace(node, node_name="second", tiles={"j": {"./j0": 0}})
    )
    with pytest.raises(ScheduleValidationError, match=r"Tile \./j0") as error:
        sched.validate()
    assert "Node: second" in str(error.value)


@pytest.mark.parametrize(
    ("terminal", "no_color", "orange"),
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_validation_error_keeps_traceback_with_optional_orange(
    monkeypatch, terminal, no_color, orange
):
    _, sch = matmul_scheduler()
    sch.tile("j", {"j0": 128, "j1": 256})
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(
        exceptions,
        "sys",
        SimpleNamespace(stderr=SimpleNamespace(isatty=lambda: terminal)),
    )

    with pytest.raises(ScheduleValidationError) as error:
        sch.schedule()
    output = StringIO()
    traceback.print_exception(error.value, file=output)
    text = output.getvalue()
    assert "Traceback (most recent call last):" in text
    assert "Invalid XTC schedule" in text
    assert "  Node: C" in text
    assert "  Root: ." in text
    assert "  Dimension: j" in text
    assert "  Error: Inner tile ./j1 (256) exceeds outer tile ./j0 (128)." in text
    assert ("\033[38;5;208m" in text) is orange
