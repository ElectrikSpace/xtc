
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


def test_mlir_compile_nonzero_pack_pad_not_implemented(tmp_path, monkeypatch):
    from xtc.backends.mlir.MlirTarget.MlirLLVMTarget import MlirLLVMTarget

    def _fake_heuristic(
        self,
        *,
        schedule_dims,
        input_idx,
        input_element_bytewidth,
        input_buffer_shape,
    ):
        del self, input_idx, input_element_bytewidth, input_buffer_shape
        return {"k": 1} if "k" in schedule_dims else {}

    monkeypatch.setattr(MlirLLVMTarget, "pack_at_padding_heuristic", _fake_heuristic)

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
    sch.unroll({"i1": 2})
    sch.pack_at("i", 1, pad=True)
    sched = sch.schedule()
    comp = impl.get_compiler(shared_lib=False, dump_file=str(tmp_path / "pack_pad"))

    with pytest.raises(NotImplementedError, match="local_buffer_at"):
        comp.compile(sched)
