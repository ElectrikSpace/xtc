import numpy as np
import pytest

import xtc.graphs.xtc.op as O
from xtc.graphs.xtc.data import XTCTensor, XTCTensorType
from xtc.kernelbase import ggml_matmul_q_graph, graph_key
from xtc.quant import GGML_QUANT_FORMATS, ggml_dequantize, ggml_quantize


@pytest.mark.parametrize("fmt", sorted(GGML_QUANT_FORMATS))
@pytest.mark.parametrize("transpose", [False, True])
def test_dequantize_types_and_forward(fmt, transpose):
    N, K = 6, 512
    qshape = GGML_QUANT_FORMATS[fmt].quantized_shape(N, K)
    rng = np.random.default_rng(0)
    q = ggml_quantize(rng.standard_normal((N, K)).astype(np.float32), fmt)
    bq = O.tensor(qshape, "uint8", name="Bq")
    with O.graph(name="deq") as gb:
        O.dequantize(bq, format=fmt, transpose=transpose, name="B")
    graph = gb.graph
    (out_type,) = graph.forward_types([XTCTensorType(qshape, "uint8")])
    assert out_type.dtype == "float32"
    assert out_type.shape == ((K, N) if transpose else (N, K))
    (out,) = graph.forward([XTCTensor(q)])
    ref = ggml_dequantize(q, fmt)
    np.testing.assert_array_equal(out.numpy(), ref.T if transpose else ref)


def test_dequantize_rejects_bad_input():
    with pytest.raises(AssertionError):
        bq = O.tensor((6, 2, 34), "uint8", name="Bq")
        with O.graph(name="deq"):
            O.dequantize(bq, format="q4_0", name="B")
    with pytest.raises(AssertionError):
        bq = O.tensor((6, 2, 18), "float32", name="Bq")
        with O.graph(name="deq"):
            O.dequantize(bq, format="q4_0", name="B")


@pytest.mark.parametrize("qtype", ["q8_0", "q4_0", "q4_K", "q6_K"])
def test_matmul_q_graph(qtype):
    M, N, K = 3, 8, 256
    graph = ggml_matmul_q_graph(M, N, K, qtype)
    rng = np.random.default_rng(1)
    a = rng.standard_normal((M, K)).astype(np.float32)
    q = ggml_quantize(rng.standard_normal((N, K)).astype(np.float32), qtype)
    (out,) = graph.forward([XTCTensor(a), XTCTensor(q)])
    np.testing.assert_allclose(
        out.numpy(), a @ ggml_dequantize(q, qtype).T, rtol=1e-5, atol=1e-5
    )

    graph_dict = graph.to_dict()
    with O.graph() as gb:
        gb.from_dict(graph_dict)
    assert gb.graph.to_dict() == graph_dict
    assert graph_key(gb.graph) == graph_key(graph)
    assert graph_key(graph) != graph_key(ggml_matmul_q_graph(M, N, K, "q8_0" if qtype == "q4_0" else "q4_0"))
