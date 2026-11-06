import pytest

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend


pytest.importorskip("mlir_sdist")


def test_mlir_pack_schedule_records_heuristic_pack_flag():
    _I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((_I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="C")
    impl = Backend(gb.graph)
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 2})
    sch.tile("j", {"j1": 16})
    sch.interchange(["k", "i", "j", "i1", "j1"])
    sch.pack_at("i", 1, pad=True)
    sched = sch.schedule()
    node = sched.schedule_impl[-1]
    assert node.packed_buffers["./i"] == [(1, True)]
