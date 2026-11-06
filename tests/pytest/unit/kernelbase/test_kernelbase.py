#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import pytest

import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.builder import graph_builder
from xtc.graphs.xtc.data import XTCTensorType
from xtc.kernelbase import DescriptSchedule, KernelBase, graph_hash, graph_key


def test_tensor_type_layout_const_roundtrip():
    t = XTCTensorType(shape=(2, 3), dtype="float32", const=True, layout=[1, 0])
    d = t.to_dict()
    assert d["const"] is True
    assert d["layout"] == [1, 0]
    t2 = XTCTensorType.from_dict(d)
    assert t2 == t


def test_graph_key_ignores_name():
    I, J, K, dtype = 4, 32, 512, "float32"

    def build(name: str):
        a = O.tensor((I, K), dtype, name="A")
        b = O.tensor((K, J), dtype, name="B")
        with O.graph(name=name) as gb:
            O.matmul(a, b, name="C")
        return gb.graph

    g1 = build("g1")
    g2 = build("g2")
    assert graph_key(g1) == graph_key(g2)
    assert graph_hash(g1) == graph_hash(g2)


def test_graph_key_sensitivity_layout():
    a1 = O.tensor((4, 512), "float32", layout=[0, 1])
    b1 = O.tensor((512, 32), "float32")
    with O.graph(name="x") as gb1:
        O.matmul(a1, b1, name="C")
    g1 = gb1.graph

    a2 = O.tensor((4, 512), "float32", layout=[1, 0])
    b2 = O.tensor((512, 32), "float32")
    with O.graph(name="x") as gb2:
        O.matmul(a2, b2, name="C")
    g2 = gb2.graph

    assert graph_hash(g1) != graph_hash(g2)


def test_descript_schedule_yaml_roundtrip(tmp_path):
    s = DescriptSchedule(
        node_name="C",
        abstract_dims=["I", "J", "K"],
        spec={"K": {}, "I": {}, "J": {}},
    )
    p = tmp_path / "s.yaml"
    s.dump(p)
    s2 = DescriptSchedule.load(p)
    assert s2.to_dict() == s.to_dict()


def test_graph_builder_graph_from_dict_roundtrip():
    I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((I, K), dtype)
    b = O.tensor((K, J), dtype)
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")
    d = gb.graph.to_dict()
    g2 = graph_builder.graph_from_dict(d)
    assert g2.to_dict() == d


@pytest.mark.skip(reason="requires MLIR toolchain; run manually or in CI with dev deps")
def test_kernelbase_add_lookup(tmp_path):
    I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")
    graph = gb.graph
    sched = DescriptSchedule(
        node_name="C",
        abstract_dims=["I", "J", "K"],
        spec={
            "K": {},
            "I": {},
            "J": {},
            "I#2": {"unroll": True},
            "J#16": {"vectorize": True},
        },
    )
    kb = KernelBase(tmp_path)
    entry = kb.add(graph, sched, overwrite=True)
    assert kb.lookup(graph) is not None
    assert entry.kernel_so.is_file()
    assert entry.test_py.is_file()


# --- canonical form -------------------------------------------------------------------


def test_canonical_keeps_operator_name():
    from xtc.kernelbase import canonical_graph_json

    a = O.tensor((4, 8), "float32")
    b = O.tensor((8, 2), "float32")
    with O.graph(name="g") as gb:
        m = O.matmul(a, b, name="mm")
        O.relu(m, name="act")
    js = canonical_graph_json(gb.graph)
    assert '"name":"matmul"' in js
    assert '"name":"relu"' in js
    # graph / node / tensor names are not part of the identity
    assert '"mm"' not in js and '"act"' not in js and '"g"' not in js


def test_graph_key_sensitivity_op():
    def build(with_relu: bool):
        a = O.tensor((4, 8), "float32")
        b = O.tensor((8, 2), "float32")
        with O.graph(name="g") as gb:
            m = O.matmul(a, b, name="mm")
            if with_relu:
                O.relu(m, name="act")
        return gb.graph

    assert graph_key(build(True)) != graph_key(build(False))


def test_graph_key_ignores_node_and_tensor_names():
    from xtc.kernelbase import ggml_matmul_graph

    g1 = ggml_matmul_graph(3, 5, 8, name="x")
    g2 = ggml_matmul_graph(3, 5, 8, name="y")
    assert graph_key(g1) == graph_key(g2)


# Golden keys. ggml-xtc (llama.cpp/ggml/src/ggml-xtc/ggml-xtc.cpp) recomputes these keys in
# C++ (own sha256 + canonical JSON string): if this test changes, the C++ side must change too.
@pytest.mark.parametrize(
    "mnk,key",
    [
        ((1, 64, 64), "bf70307699bdf9ce"),
        ((1, 32, 64), "8e3a658eb1a759f1"),
        ((1, 128, 64), "04b12abd89ea9d99"),
        ((1, 64, 128), "0cc62dddf299374d"),
        ((1, 259, 64), "5900679f9e3fbbff"),
    ],
)
def test_ggml_matmul_golden_key(mnk, key):
    from xtc.kernelbase import ggml_matmul_graph

    assert graph_key(ggml_matmul_graph(*mnk)) == key


def test_ggml_matmul_canonical_json_golden():
    from xtc.kernelbase import canonical_graph_json, ggml_matmul_graph

    assert canonical_graph_json(ggml_matmul_graph(3, 5, 8)) == (
        '{"inputs":[{"expr":{"type":{"dtype":"float32","shape":[3,8]}},"uid":"%0"},'
        '{"expr":{"type":{"dtype":"float32","layout":[1,0],"shape":[8,5]}},"uid":"%1"}],'
        '"nodes":[{"expr":{"args":["%0","%1"],"op":{"attrs":{},"name":"matmul"}},"uid":"%2"}],'
        '"outputs":[{"uid":"%2"}]}'
    )


# --- pending stubs (written by llama.cpp) -> build --------------------------------------

_STUB_GRAPH = """\
name: matmul
ops_version: unknown
inputs:
- uid: matmul_in0
  name: A
  expr:
    type: {shape: [1, 16], dtype: float32}
- uid: matmul_in1
  name: B
  expr:
    type: {shape: [16, 8], dtype: float32, layout: [1, 0]}
nodes:
- uid: matmul_node0
  name: C
  expr:
    op: {name: matmul, attrs: {}}
    args: [matmul_in0, matmul_in1]
outputs:
- uid: matmul_node0
"""

_STUB_SCHEDULE = """\
{todo}node_name: C
abstract_dims: [i, j, k]
default_node: null
spec:
  i: {{}}
  j: {{}}
  k: {{}}
"""


def _make_stub(root, todo: bool):
    from xtc.kernelbase import ggml_matmul_graph

    key = graph_key(ggml_matmul_graph(1, 8, 16))
    d = root / "_pending" / key
    d.mkdir(parents=True)
    (d / "graph.yaml").write_text(_STUB_GRAPH)
    (d / "schedule.yaml").write_text(_STUB_SCHEDULE.format(todo="todo: true\n" if todo else ""))
    return key, d


def test_stub_graph_yaml_has_the_ggml_key(tmp_path):
    """The graph.yaml written by C++ must hash to the key it was filed under."""
    key, d = _make_stub(tmp_path, todo=True)
    assert graph_key(graph_builder.graph_from_file(d / "graph.yaml")) == key


def test_pending_and_build_refuse_todo(tmp_path, capsys):
    from xtc.cli import kernelbase as cli

    key, d = _make_stub(tmp_path, todo=True)
    kb = KernelBase(tmp_path)
    assert cli._pending_keys(kb) == [key]
    assert cli._stub_is_todo(d)
    assert cli._build_one(kb, key, force=False, overwrite=False) is False
    assert kb.keys() == []  # nothing inserted, stub untouched
    assert d.is_dir()


def test_build_rejects_modified_graph(tmp_path):
    from xtc.cli import kernelbase as cli

    key, d = _make_stub(tmp_path, todo=False)
    (d / "graph.yaml").write_text(_STUB_GRAPH.replace("[16, 8]", "[16, 9]"))
    kb = KernelBase(tmp_path)
    assert cli._build_one(kb, key, force=False, overwrite=False) is False
    assert kb.keys() == []


def test_build_pending_stub_end_to_end(tmp_path):
    pytest.importorskip("mlir")
    from xtc.cli import kernelbase as cli

    key, d = _make_stub(tmp_path, todo=False)
    kb = KernelBase(tmp_path)
    assert cli._build_one(kb, key, force=False, overwrite=False) is True
    assert kb.keys() == [key]
    assert kb.get(key).kernel_so.is_file()
    assert not d.exists()  # stub consumed


# --- script-style stubs (build_kernel.py, written by ggml-xtc) ---------------------------

_STUB_SCRIPT = """\
from xtc.kernelbase.pending import run_stub

TODO = {todo}
NODE_NAME = "C"
ABSTRACT_DIMS = ["i", "j", "k"]
SPEC = {{"i": {{}}, "j": {{}}, "k": {{}}}}

if __name__ == "__main__":
    raise SystemExit(run_stub(__file__, NODE_NAME, ABSTRACT_DIMS, SPEC, todo=TODO))
"""


def _make_script_stub(root, todo: bool):
    from xtc.kernelbase import ggml_matmul_graph

    key = graph_key(ggml_matmul_graph(1, 8, 16))
    d = root / "_pending" / key
    d.mkdir(parents=True)
    (d / "graph.yaml").write_text(_STUB_GRAPH)
    (d / "build_kernel.py").write_text(_STUB_SCRIPT.format(todo=todo))
    return key, d


def test_script_stub_todo_detection(tmp_path):
    from xtc.kernelbase import pending

    key, d = _make_script_stub(tmp_path, todo=True)
    kb = KernelBase(tmp_path)
    assert pending.pending_keys(kb) == [key]
    assert pending.stub_is_todo(d)
    (d / "build_kernel.py").write_text(_STUB_SCRIPT.format(todo=False))
    assert not pending.stub_is_todo(d)


def test_script_stub_refuses_todo(tmp_path):
    import subprocess
    import sys

    key, d = _make_script_stub(tmp_path, todo=True)
    r = subprocess.run([sys.executable, str(d / "build_kernel.py")], capture_output=True, text=True)
    assert r.returncode == 2
    assert "schedule not provided" in r.stderr
    assert KernelBase(tmp_path).keys() == []
    assert d.is_dir()


def test_script_stub_end_to_end(tmp_path):
    pytest.importorskip("mlir")
    import subprocess
    import sys

    key, d = _make_script_stub(tmp_path, todo=False)
    # root is inferred from <root>/_pending/<key>/build_kernel.py
    r = subprocess.run([sys.executable, str(d / "build_kernel.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kb = KernelBase(tmp_path)
    assert kb.keys() == [key]
    assert kb.get(key).kernel_so.is_file()
    assert not d.exists()  # stub consumed


def test_cli_build_runs_script_stub(tmp_path):
    pytest.importorskip("mlir")
    from xtc.cli import kernelbase as cli

    key, d = _make_script_stub(tmp_path, todo=True)
    kb = KernelBase(tmp_path)
    assert cli._build_one(kb, key, force=False, overwrite=False) is False
    assert cli._build_one(kb, key, force=True, overwrite=False) is True
    assert kb.keys() == [key]
