import pytest

from xtc.quant import ggml_quant_format, q4_k_m_tensor_format

# Formats chosen by llama-quantize Q4_K_M for a 16 layers llama model:
# q6_K for the attn_v and ffn_down weights of these layers, q4_K otherwise.
N_LAYER = 16
MORE_BITS_LAYERS = {0, 1, 4, 7, 10, 13, 14, 15}


@pytest.mark.parametrize("tensor", ["attn_v", "ffn_down"])
def test_more_bits_layers(tensor):
    formats = {
        i: q4_k_m_tensor_format(f"blk.{i}.{tensor}.weight", 4096, N_LAYER)
        for i in range(N_LAYER)
    }
    assert {i for i, fmt in formats.items() if fmt == "q6_K"} == MORE_BITS_LAYERS
    assert {fmt for fmt in formats.values()} == {"q4_K", "q6_K"}


@pytest.mark.parametrize(
    "name", ["attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"]
)
def test_q4_k_weights(name):
    for i in range(N_LAYER):
        assert q4_k_m_tensor_format(f"blk.{i}.{name}.weight", 4096, N_LAYER) == "q4_K"


def test_embeddings_and_output():
    assert q4_k_m_tensor_format("token_embd.weight", 4096, N_LAYER) == "q4_K"
    assert q4_k_m_tensor_format("output.weight", 4096, N_LAYER) == "q6_K"
    tied = q4_k_m_tensor_format("token_embd.weight", 4096, N_LAYER, True)
    assert tied == "q6_K"
    assert q4_k_m_tensor_format("blk.0.attn_qkv.weight", 4096, N_LAYER) == "q5_K"


def test_row_size_fallbacks():
    # rows of 640 values are not made of 256 values blocks
    assert q4_k_m_tensor_format("blk.2.ffn_down.weight", 640, N_LAYER) == "q5_0"
    assert q4_k_m_tensor_format("blk.0.ffn_down.weight", 640, N_LAYER) == "q8_0"
    assert q4_k_m_tensor_format("output.weight", 640, N_LAYER) == "q8_0"
    assert q4_k_m_tensor_format("blk.0.attn_qkv.weight", 640, N_LAYER) == "q5_1"
    assert q4_k_m_tensor_format("blk.0.attn_q.weight", 100, N_LAYER) == "f16"


def test_formats_are_supported():
    # all the formats of a Q4_K_M model with standard row sizes are supported
    names = ["token_embd", "output"] + [
        f"blk.{i}.{t}"
        for i in range(N_LAYER)
        for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_down")
    ]
    for ncols in (4096, 640):
        for name in names:
            ggml_quant_format(q4_k_m_tensor_format(f"{name}.weight", ncols, N_LAYER))
