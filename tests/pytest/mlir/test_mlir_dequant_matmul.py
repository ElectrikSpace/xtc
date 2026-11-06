import numpy as np
import pytest
from pathlib import Path
from mlir_utils import requires_mlir

M, N = 4, 32
FORMATS = ("q8_0", "q4_0", "q5_0", "q4_K", "q5_K", "q6_K")


def reduction_size(qtype):
    from xtc.quant import ggml_quant_format

    # two blocks of the 256 values formats
    return 128 if ggml_quant_format(qtype).qk == 32 else 512


def dequant_matmul_impl(qtype, use_tensor_dialect=True):
    from xtc.backends.mlir import Backend
    from xtc.kernelbase import ggml_matmul_q_graph

    return Backend(
        ggml_matmul_q_graph(M, N, reduction_size(qtype), qtype),
        use_tensor_dialect=use_tensor_dialect,
    )


def schedule(impl, mode, k_tile=32):
    sch = impl.get_scheduler()
    sch.tile("j", {"j1": 16})
    sch.tile("k", {"k1": k_tile})
    sch.interchange(["i", "j", "k", "k1", "j1"])
    if mode == "producer":
        sch.fuse_producer_at("k", 1)
    elif mode == "pack":
        sch.pack_at("k", 1)
    elif mode == "local_pack":
        sch.pack_at("k", 1, mtype="local")
    sch.vectorize(["j1"])
    return sch.schedule()


def compile_and_run(impl, sched, qtype, tmpdir, **compiler_args):
    from xtc.quant import ggml_dequantize, ggml_quantize

    compiler = impl.get_compiler(
        dump_file=str(Path(tmpdir) / impl.graph.name),
        shared_lib=True,
        **compiler_args,
    )
    module = compiler.compile(sched)

    # built-in validation against the graph reference on generated inputs
    results, code, error_msg = module.get_evaluator(validate=True).evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"

    # Realistic weights: outputs come from sums of K terms of both signs, the
    # tolerance accounts for the different float32 accumulation orders.
    K = reduction_size(qtype)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float32)
    bq = ggml_quantize(rng.standard_normal((N, K)).astype(np.float32), qtype)
    c = np.zeros((M, N), np.float32)
    results, code, error_msg = module.get_evaluator(
        parameters=([a, bq], [c])
    ).evaluate()
    assert code == 0, f"failed to evaluate: {error_msg}"
    ref = a @ ggml_dequantize(bq, qtype).T
    np.testing.assert_allclose(c, ref, rtol=1e-5, atol=1e-4)


@requires_mlir
@pytest.mark.parametrize("qtype", FORMATS)
@pytest.mark.parametrize("mode", ["none", "producer", "pack", "local_pack"])
def test_dequant_matmul(tmpdir, capfd, qtype, mode):
    from xtc.quant import ggml_quant_format

    impl = dequant_matmul_impl(qtype)
    sched = schedule(impl, mode)
    compile_and_run(impl, sched, qtype, tmpdir, print_bufferization_ir=True)
    ir = capfd.readouterr().err.split("IR Dump After Tensor Lowering")[-1]
    allocs = [line for line in ir.splitlines() if "memref.alloc" in line]
    dequantizing_pack = mode in ("pack", "local_pack")
    # A fused dequantize op, as for a dequantizing pack, writes the dequantized
    # tile (k1 x j1) to a local buffer, otherwise the whole weight is computed.
    K = reduction_size(qtype)
    assert any("memref<32x16xf32>" in line for line in allocs) == (mode != "none")
    assert any(f"memref<{K}x32xf32>" in line for line in allocs) == (mode == "none")
    # The staged pack copies the bytes of the quantized blocks read by the
    # tile to local buffers, one per field of the format, e.g. for the 32
    # values of a tile of q4_K: 7 bytes of scales and 32 bytes of quants.
    staged = [line for line in allocs if "xi8>" in line]
    fmt = ggml_quant_format(qtype)
    assert len(staged) == (len(fmt.fields) if mode == "local_pack" else 0)
    if mode == "local_pack":
        tile_bytes = [field.index(31) + 1 for field in fmt.fields]
        assert sorted(f"16x1x{size}xi8" for size in tile_bytes) == sorted(
            line.split("memref<")[-1].rstrip(">") for line in staged
        )
    if dequantizing_pack:
        # tile buffers are allocated once, out of the loops
        assert all("memref.alloca" in line for line in allocs)
        assert ir.index("memref<32x16xf32>") < ir.index("scf.for")


@requires_mlir
@pytest.mark.parametrize("qtype", FORMATS)
def test_dequant_matmul_memref(tmpdir, qtype):
    impl = dequant_matmul_impl(qtype, use_tensor_dialect=False)
    compile_and_run(impl, schedule(impl, "none"), qtype, tmpdir)


@requires_mlir
def test_dequant_pack_requires_tensor_dialect():
    impl = dequant_matmul_impl("q4_0", use_tensor_dialect=False)
    with pytest.raises(NotImplementedError):
        schedule(impl, "pack")


@requires_mlir
@pytest.mark.parametrize(
    "qtype,k_tile,valid",
    [
        ("q4_0", 16, True),
        ("q4_0", 64, True),
        ("q4_0", 48, False),
        ("q4_K", 128, True),
        ("q4_K", 96, False),
    ],
)
def test_dequant_pack_block_alignment(qtype, k_tile, valid):
    impl = dequant_matmul_impl(qtype)
    if valid:
        schedule(impl, "local_pack", k_tile=k_tile)
    else:
        with pytest.raises(ValueError):
            schedule(impl, "local_pack", k_tile=k_tile)


@requires_mlir
@pytest.mark.parametrize("qtype", ["q4_K", "q6_K"])
@pytest.mark.parametrize("k_tile", [16, 64, 256, 512])
def test_dequant_pack_k_tiles(tmpdir, qtype, k_tile):
    # tiles smaller than, as large as and larger than the 256 values blocks
    impl = dequant_matmul_impl(qtype)
    compile_and_run(impl, schedule(impl, "local_pack", k_tile=k_tile), qtype, tmpdir)
