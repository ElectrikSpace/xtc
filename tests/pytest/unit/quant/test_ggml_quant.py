import hashlib

import numpy as np
import pytest

from xtc.quant import (
    GGML_QUANT_FORMATS,
    ggml_dequantize,
    ggml_quant_format,
    ggml_quantize,
)

FORMATS = sorted(GGML_QUANT_FORMATS)

# Reference bytes produced by llama.cpp gguf-py (gguf.quants.quantize) for
# x = (arange(64) - 20.5) / 7, i.e. one row of two blocks.
GOLDEN_X = (np.arange(64, dtype=np.float32).reshape(1, 64) - 20.5) / 7.0
GOLDEN = {
    "q8_0": [
        231, 37, 129, 135, 141, 148, 154, 160, 166, 172, 179, 185, 191, 197,
        203, 210, 216, 222, 228, 234, 241, 247, 253, 3, 9, 15, 22, 28, 34, 40,
        46, 53, 59, 65, 31, 42, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64,
        67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112,
        115, 118, 121, 124, 127,
    ],
    "q4_0": [
        219, 53, 96, 112, 113, 113, 130, 130, 146, 147, 147, 164, 164, 180,
        181, 181, 197, 198, 18, 186, 54, 54, 37, 37, 37, 37, 37, 37, 20, 20,
        20, 20, 20, 3, 3, 3,
    ],
}

# sha256 of the blocks and of their dequantization by the C reference
# (quantize_row_*_ref and dequantize_row_* of ggml/src/ggml-quants.c) for the
# 2 rows of 512 values of GOLDEN_KX, computed exactly in float32.
_i = np.arange(2 * 512)
GOLDEN_KX = (
    ((((_i * 37) % 101) - 50) * (1 + (_i // 32) % 5) / np.float32(7))
    .astype(np.float32)
    .reshape(2, 512)
)
GOLDEN_SHA256 = {
    "q5_0": (
        "37e22c8ba478d9c23d80579b02d555cf2ab4b188bfdc659b4e1a0a2b9e340323",
        "170d1aec1aae89120f9ecc8a9783a2c31295e1255005b3b4717c0d1b398c4c24",
    ),
    "q4_K": (
        "1b56c7c12bc41242fc39925fe90f9e06a9c58bb685c45d75acbc000c24cc2eb2",
        "29b8f26de6f560d6943e3d72a4aedfbcaa5e2bd405a5bea47bb60b3fc284ac48",
    ),
    "q5_K": (
        "fa82202cae25ea67d96ab7c196f50419e50b40a142c1092cb0392833dce99ca0",
        "3fc812ca76dcc34e4dc0bc934a9431233e411f1e1a7d23c5bc74c04630ea6a37",
    ),
    "q6_K": (
        "ef5568ea1493e85ba9ff27d4273898079ecd70ea39a6401034d8d1eb3e4f32ef",
        "c04949ec233654c36abd2a131570f5287fa296dbe9416f97d61b77718e04a67a",
    ),
}

# relative RMS quantization error of standard normal values
MAX_RELATIVE_ERROR = {
    "q8_0": 0.01,
    "q6_K": 0.025,
    "q5_K": 0.045,
    "q5_0": 0.05,
    "q4_K": 0.08,
    "q4_0": 0.095,
}


def f16_bytes(v):
    return np.array([v], "<f2").view(np.uint8)


@pytest.mark.parametrize("fmt", sorted(GOLDEN))
def test_quantize_matches_ggml_reference(fmt):
    q = ggml_quantize(GOLDEN_X, fmt)
    assert q.shape == (1, 2, ggml_quant_format(fmt).block_bytes)
    assert q.dtype == np.uint8
    assert q.reshape(-1).tolist() == GOLDEN[fmt]


@pytest.mark.parametrize("fmt", sorted(GOLDEN_SHA256))
def test_codec_matches_ggml_c_reference(fmt):
    q = ggml_quantize(GOLDEN_KX, fmt)
    fmt_ = ggml_quant_format(fmt)
    assert q.shape == (2, 512 // fmt_.qk, fmt_.block_bytes)
    q_sha, y_sha = GOLDEN_SHA256[fmt]
    assert hashlib.sha256(q.tobytes()).hexdigest() == q_sha
    y = ggml_dequantize(q, fmt)
    assert y.shape == (2, 512) and y.dtype == np.float32
    assert hashlib.sha256(y.tobytes()).hexdigest() == y_sha


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_error(fmt):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3, 512)).astype(np.float32) * 4
    q = ggml_quantize(x, fmt)
    fmt_ = ggml_quant_format(fmt)
    assert q.shape == (5, 3, 512 // fmt_.qk, fmt_.block_bytes)
    y = ggml_dequantize(q, fmt)
    assert y.shape == x.shape and y.dtype == np.float32
    rel = np.sqrt(np.mean((x - y) ** 2) / np.mean(x**2))
    assert rel < MAX_RELATIVE_ERROR[fmt]
    if fmt == "q5_K":
        return  # the scales search of q5_K may find other scales
    # dequantized values are quantized to the same values
    np.testing.assert_array_equal(ggml_dequantize(ggml_quantize(y, fmt), fmt), y)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
def test_roundtrip_error_bound(fmt):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3, 128)).astype(np.float32) * 4
    q = ggml_quantize(x, fmt)
    y = ggml_dequantize(q, fmt)
    # Per block error is bounded by half a quantization step (a full one for
    # q4_0, which clamps +8 to 7), plus the fp16 rounding of the scale d used
    # for each of the (at most 2**(bits-1)) steps.
    d = np.abs(q[..., :2].copy().view("<f2")[..., 0].astype(np.float32))
    err = np.abs(x - y).reshape(5, 3, 4, 32).max(axis=-1)
    bits = ggml_quant_format(fmt).bits
    bound = d * ((0.5 if fmt == "q8_0" else 1.0) + 2 ** (bits - 1) * 2.0**-11)
    assert np.all(err <= bound + 1e-6)


@pytest.mark.parametrize("fmt", FORMATS)
def test_zero_block(fmt):
    q = ggml_quantize(np.zeros((2, 256), np.float32), fmt)
    np.testing.assert_array_equal(ggml_dequantize(q, fmt), np.zeros((2, 256)))


def test_q4_0_nibble_order():
    # value j is the low nibble of qs[j], value j + 16 its high nibble
    block = np.zeros(18, np.uint8)
    block[:2] = f16_bytes(2.0)
    block[2:] = np.arange(16, dtype=np.uint8) | (np.uint8(15) - np.arange(16, dtype=np.uint8)) << 4
    y = ggml_dequantize(block[None, :], "q4_0")
    expected = 2.0 * (np.concatenate([np.arange(16), 15 - np.arange(16)]) - 8)
    np.testing.assert_array_equal(y, expected)


def test_q8_0_signed_values():
    block = np.zeros(34, np.uint8)
    block[:2] = f16_bytes(0.5)
    block[2:] = np.arange(-16, 16, dtype=np.int8).view(np.uint8)
    y = ggml_dequantize(block[None, :], "q8_0")
    np.testing.assert_array_equal(y, 0.5 * np.arange(-16, 16))


def test_q4_k_layout():
    # 8 sub-blocks j of 32 values: x = d * sc[j] * q - dmin * m[j], with
    # 6 bit sc/m packed in 12 bytes, the high bits of sub-blocks 4..7 in the
    # top bits of the bytes of sub-blocks 0..3
    sc = np.array([1, 2, 3, 4, 40, 50, 60, 63], np.uint8)
    m = np.array([5, 6, 7, 8, 33, 44, 55, 62], np.uint8)
    L = (np.arange(256) * 7 % 16).astype(np.uint8)
    block = np.zeros(144, np.uint8)
    block[0:2] = f16_bytes(0.5)
    block[2:4] = f16_bytes(0.25)
    block[4:8] = sc[:4] | ((sc[4:] >> 4) << 6)
    block[8:12] = m[:4] | ((m[4:] >> 4) << 6)
    block[12:16] = (sc[4:] & 15) | ((m[4:] & 15) << 4)
    # value 64 * g + l: low nibble of qs[32 * g + l] for l < 32, else high
    L4 = L.reshape(4, 2, 32)
    block[16:] = (L4[:, 0] | (L4[:, 1] << 4)).reshape(-1)
    y = ggml_dequantize(block[None, None, :], "q4_k")
    j = np.arange(256) // 32
    expected = np.float32(0.5) * sc[j] * L - np.float32(0.25) * m[j]
    np.testing.assert_array_equal(y[0], expected.astype(np.float32))


def test_q6_k_layout():
    # 16 sub-blocks j of 16 values: x = d * sc[j] * (q - 32), sc signed
    sc = np.arange(-8, 8, dtype=np.int8)
    L = (np.arange(256) * 5 % 64).astype(np.uint8)
    block = np.zeros(210, np.uint8)
    # in each half of 128 values, value 32 * r + l: low 4 bits in the nibble
    # r // 2 of ql[32 * (r % 2) + l], high 2 bits in bits 2 * r of qh[l]
    L6 = L.reshape(2, 4, 32)
    ql = (L6[:, 0:2] & 15) | ((L6[:, 2:4] & 15) << 4)
    qh = sum((L6[:, r] >> 4) << (2 * r) for r in range(4))
    block[0:128] = ql.reshape(-1)
    block[128:192] = qh.reshape(-1)
    block[192:208] = sc.view(np.uint8)
    block[208:210] = f16_bytes(0.125)
    y = ggml_dequantize(block[None, :], "q6_K")
    expected = np.float32(0.125) * sc[np.arange(256) // 16] * (L.astype(np.int32) - 32)
    np.testing.assert_array_equal(y, expected.astype(np.float32))


def test_format_names():
    assert ggml_quant_format("q4_k") is ggml_quant_format("q4_K")
    assert ggml_quant_format("q6_K").block_bytes == 210
    with pytest.raises(ValueError):
        ggml_quant_format("q2_K")
    with pytest.raises(AssertionError):
        ggml_quantize(np.zeros((1, 48), np.float32), "q4_0")
    with pytest.raises(AssertionError):
        ggml_quantize(np.zeros((1, 384), np.float32), "q4_K")
    with pytest.raises(AssertionError):
        ggml_quant_format("q8_0").quantized_shape(4, 33)
