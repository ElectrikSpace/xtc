import numpy as np
import pytest
from pathlib import Path
from tvm_utils import requires_tvm

M, N = 4, 32
FORMATS = ("q8_0", "q4_0", "q5_0", "q4_K", "q5_K", "q6_K")


def reduction_size(qtype):
    from xtc.quant import ggml_quant_format

    # two blocks of the 256 values formats
    return 128 if ggml_quant_format(qtype).qk == 32 else 512


def dequant_matmul_impl(qtype):
    from xtc.backends.tvm import TVMBackend
    from xtc.kernelbase import ggml_matmul_q_graph

    return TVMBackend(ggml_matmul_q_graph(M, N, reduction_size(qtype), qtype))


def schedule(impl, mode):
    sch = impl.get_scheduler()
    sch.tile("j", {"j1": 16})
    sch.tile("k", {"k1": 32})
    sch.interchange(["i", "j", "k", "k1", "j1"])
    if mode == "producer":
        sch.fuse_producer_at("k", 1)
    elif mode == "pack":
        sch.pack_at("k", 1)
    elif mode == "local_pack":
        sch.pack_at("k", 1, mtype="local")
    sch.vectorize(["j1"])
    return sch.schedule()


def quantized_parameters(qtype):
    from xtc.quant import ggml_dequantize, ggml_quantize

    K = reduction_size(qtype)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    bq = ggml_quantize(rng.standard_normal((N, K)).astype(np.float32), qtype)
    ref = a @ ggml_dequantize(bq, qtype).T
    return a, bq, ref


def compile_and_run(impl, sched, qtype, tmpdir, **compiler_args):
    compiler = impl.get_compiler(
        dump_file=str(Path(tmpdir) / impl.graph.name), **compiler_args
    )
    module = compiler.compile(sched)

    # built-in validation against the graph reference on generated inputs
    results, code, error_msg = module.get_evaluator(validate=True).evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"

    # Realistic weights: outputs come from sums of K terms of both signs, the
    # tolerance accounts for the different float32 accumulation orders.
    a, bq, ref = quantized_parameters(qtype)
    c = np.zeros((M, N), np.float32)
    evaluator = module.get_evaluator(parameters=([a, bq], [c]))
    results, code, error_msg = evaluator.evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"
    np.testing.assert_allclose(c, ref, rtol=1e-5, atol=1e-4)


@requires_tvm
@pytest.mark.parametrize("qtype", FORMATS)
@pytest.mark.parametrize("mode", ["none", "producer", "pack", "local_pack"])
def test_dequant_matmul(tmpdir, capsys, qtype, mode):
    impl = dequant_matmul_impl(qtype)
    sched = schedule(impl, mode)
    sched_str = str(sched)
    dequantizing_pack = mode in ("pack", "local_pack")
    assert ('compute_inline(sch.get_sblock("B"))' in sched_str) == dequantizing_pack
    assert ("cache_read(I_DQ0, 0" in sched_str) == (mode == "local_pack")

    compile_and_run(impl, sched, qtype, tmpdir, shared_lib=True, print_transformed_ir=True)
    ir = capsys.readouterr().out
    # The dequantize stage remains a separate block, over the whole weight or
    # computed at k, unless it is moved into the pack.
    assert ('T.sblock("B")' in ir) == (not dequantizing_pack)
    if dequantizing_pack:
        # the pack of the matmul operand decodes the quantized blocks
        assert 'T.sblock("B_global")' in ir
    # the staged pack first copies the raw blocks to a local uint8 buffer
    allocs = [line for line in ir.splitlines() if "sblock_alloc_buffer" in line]
    assert any('"uint8"' in line for line in allocs) == (mode == "local_pack")


@requires_tvm
@pytest.mark.parametrize("module_type", ["shared_lib", "emit_c", "ar_lib"])
@pytest.mark.parametrize("bare_ptr", [False, True])
def test_dequant_matmul_backend_variant(tmpdir, module_type, bare_ptr):
    impl = dequant_matmul_impl("q4_0")
    sched = schedule(impl, "local_pack")
    compiler_args = {module_type: True, "bare_ptr": bare_ptr}
    compile_and_run(impl, sched, "q4_0", tmpdir, **compiler_args)
    if bare_ptr and module_type == "emit_c":
        wrapper = next(Path(tmpdir).glob("*.c")).read_text()
        assert "const uint8_t *input1" in wrapper
        assert "DL_input1_dtype = { 1, 8, 1 }" in wrapper
