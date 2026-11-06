#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""ggml (llama.cpp) block quantization formats.

A ggml quantized row of ``K`` values is a sequence of ``K / qk`` blocks laid
out back to back, each block holding its scale(s) and the packed quants (see
``ggml/src/ggml-common.h``). XTC keeps this byte layout as is, so that a
kernel can be called directly on a ggml weight buffer: a quantized tensor of
``rows`` rows is a ``uint8`` tensor of shape ``(rows, K // qk, block_bytes)``.

A format is described by the bytes each value of a block reads (``fields``,
at offsets that are affine in the index of the value in the block) and by a
``decode`` function of these bytes. The description is evaluated with NumPy
here and lowered by the backends (TVM expressions, MLIR affine maps and arith
ops), such that all implementations decode the same way.

The NumPy codecs below are bit-exact with the ggml reference implementations
(``quantize_row_*_ref`` / ``dequantize_row_*`` in ``ggml/src/ggml-quants.c``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing

__all__ = [
    "GGML_QUANT_FORMATS",
    "GGMLDecodeOps",
    "GGMLField",
    "GGMLQuantFormat",
    "ggml_dequantize",
    "ggml_quant_format",
    "ggml_quantize",
]


@dataclass(frozen=True)
class GGMLField:
    """A byte read by each value of a block.

    Value ``e`` of a block (``0 <= e < qk``) reads the byte at offset
    ``offset + index(e)`` of the block, with ``0 <= index(e) < size``.
    ``index`` only uses ``+``, ``*``, ``//`` and ``%`` with integer constants,
    such that it is an affine expression of ``e``.
    """

    offset: int
    size: int = 1
    index: Callable[[Any], Any] = lambda e: 0


class GGMLDecodeOps(Protocol):
    """Operations available to ``GGMLQuantFormat.decode``.

    Besides these, the integer values (``e`` and the bytes, zero extended)
    support ``+ - * & | << >>``, ``e`` also ``//`` and ``%`` by constants,
    and the float values ``+ - *``.
    """

    def f32(self, x: Any) -> Any:
        """float32 value of the integer ``x``."""
        ...

    def fp16(self, lo: Any, hi: Any) -> Any:
        """float32 value of the little endian fp16 bytes ``lo``, ``hi``."""
        ...


@dataclass(frozen=True)
class GGMLQuantFormat:
    """Layout of a ggml quantization block.

    Attributes:
        name: ggml type name, e.g. ``"q4_0"``.
        qk: number of values per block (along the reduction dimension).
        block_bytes: size in bytes of a block.
        bits: number of bits per quant.
        fields: bytes of the block read by each value.
        decode: ``decode(ops, e, *bytes)`` is the value ``e`` of the block,
            given the bytes read for ``fields``.
    """

    name: str
    qk: int
    block_bytes: int
    bits: int
    fields: tuple[GGMLField, ...]
    decode: Callable[..., Any]

    def quantized_shape(self, rows: int, k: int) -> tuple[int, int, int]:
        assert k % self.qk == 0, (
            f"{self.name}: row size {k} is not a multiple of the block size {self.qk}"
        )
        return (rows, k // self.qk, self.block_bytes)

    def dequantized_size(self, nblocks: int) -> int:
        return nblocks * self.qk


def _sext8(b: Any) -> Any:
    return b - ((b >> 7) << 8)


def _decode_q8_0(ops: GGMLDecodeOps, e: Any, d_lo: Any, d_hi: Any, q: Any) -> Any:
    return ops.f32(_sext8(q)) * ops.fp16(d_lo, d_hi)


def _decode_q4_0(ops: GGMLDecodeOps, e: Any, d_lo: Any, d_hi: Any, q: Any) -> Any:
    nibble = (q >> ((e // 16) * 4)) & 15
    return ops.f32(nibble - 8) * ops.fp16(d_lo, d_hi)


def _decode_q5_0(
    ops: GGMLDecodeOps, e: Any, d_lo: Any, d_hi: Any, qh: Any, q: Any
) -> Any:
    nibble = (q >> ((e // 16) * 4)) & 15
    high = (qh >> (e % 8)) & 1
    return ops.f32(nibble + (high << 4) - 16) * ops.fp16(d_lo, d_hi)


def _k4_scale_min(e: Any, s0: Any, s1: Any, s2: Any) -> tuple[Any, Any]:
    # get_scale_min_k4(j = e // 32) with s0, s1, s2 = scales[j % 4 + 0, 4, 8]
    hi = e // 128
    lo = 1 - hi
    sc = lo * (s0 & 63) + hi * ((s2 & 15) | ((s0 >> 6) << 4))
    m = lo * (s1 & 63) + hi * ((s2 >> 4) | ((s1 >> 6) << 4))
    return sc, m


def _decode_q4_k(
    ops: GGMLDecodeOps,
    e: Any,
    d_lo: Any,
    d_hi: Any,
    m_lo: Any,
    m_hi: Any,
    s0: Any,
    s1: Any,
    s2: Any,
    q: Any,
) -> Any:
    sc, m = _k4_scale_min(e, s0, s1, s2)
    nibble = (q >> (((e // 32) % 2) * 4)) & 15
    d = ops.fp16(d_lo, d_hi) * ops.f32(sc)
    return d * ops.f32(nibble) - ops.fp16(m_lo, m_hi) * ops.f32(m)


def _decode_q5_k(
    ops: GGMLDecodeOps,
    e: Any,
    d_lo: Any,
    d_hi: Any,
    m_lo: Any,
    m_hi: Any,
    s0: Any,
    s1: Any,
    s2: Any,
    qh: Any,
    q: Any,
) -> Any:
    sc, m = _k4_scale_min(e, s0, s1, s2)
    nibble = (q >> (((e // 32) % 2) * 4)) & 15
    high = (qh >> (e // 32)) & 1
    d = ops.fp16(d_lo, d_hi) * ops.f32(sc)
    return d * ops.f32(nibble + (high << 4)) - ops.fp16(m_lo, m_hi) * ops.f32(m)


def _decode_q6_k(
    ops: GGMLDecodeOps, e: Any, ql: Any, qh: Any, sc: Any, d_lo: Any, d_hi: Any
) -> Any:
    nibble = (ql >> (((e // 64) % 2) * 4)) & 15
    high = (qh >> (((e // 32) % 4) * 2)) & 3
    d = ops.fp16(d_lo, d_hi) * ops.f32(_sext8(sc))
    return d * ops.f32(nibble + (high << 4) - 32)


_F16_LO, _F16_HI = GGMLField(0), GGMLField(1)
# scales[12] of the q4_K/q5_K blocks at offset 4, see _k4_scale_min
_K4_SCALES = tuple(GGMLField(4 + 4 * i, 4, lambda e: (e // 32) % 4) for i in range(3))

# struct block_q8_0 { ggml_half d; int8_t qs[32]; }
Q8_0 = GGMLQuantFormat(
    name="q8_0",
    qk=32,
    block_bytes=34,
    bits=8,
    fields=(_F16_LO, _F16_HI, GGMLField(2, 32, lambda e: e)),
    decode=_decode_q8_0,
)
# struct block_q4_0 { ggml_half d; uint8_t qs[16]; }, value j in the low nibble
# of qs[j] and value j + 16 in the high nibble of qs[j]
Q4_0 = GGMLQuantFormat(
    name="q4_0",
    qk=32,
    block_bytes=18,
    bits=4,
    fields=(_F16_LO, _F16_HI, GGMLField(2, 16, lambda e: e % 16)),
    decode=_decode_q4_0,
)
# struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[16]; }, low bits
# as q4_0, bit 4 of value j is bit j of qh
Q5_0 = GGMLQuantFormat(
    name="q5_0",
    qk=32,
    block_bytes=22,
    bits=5,
    fields=(
        _F16_LO,
        _F16_HI,
        GGMLField(2, 4, lambda e: e // 8),
        GGMLField(6, 16, lambda e: e % 16),
    ),
    decode=_decode_q5_0,
)
# struct block_q4_K { ggml_half d, dmin; uint8_t scales[12]; uint8_t qs[128]; }
# 8 sub-blocks of 32 values with 6 bit scales and mins, value 64 * g + l in the
# low nibble of qs[32 * g + l] for l < 32, else in its high nibble
Q4_K = GGMLQuantFormat(
    name="q4_K",
    qk=256,
    block_bytes=144,
    bits=4,
    fields=(
        _F16_LO,
        _F16_HI,
        GGMLField(2),
        GGMLField(3),
        *_K4_SCALES,
        GGMLField(16, 128, lambda e: 32 * (e // 64) + e % 32),
    ),
    decode=_decode_q4_k,
)
# struct block_q5_K { ggml_half d, dmin; uint8_t scales[12]; uint8_t qh[32];
# uint8_t qs[128]; }, low bits as q4_K, bit 4 of value e is bit e // 32 of
# qh[e % 32]
Q5_K = GGMLQuantFormat(
    name="q5_K",
    qk=256,
    block_bytes=176,
    bits=5,
    fields=(
        _F16_LO,
        _F16_HI,
        GGMLField(2),
        GGMLField(3),
        *_K4_SCALES,
        GGMLField(16, 32, lambda e: e % 32),
        GGMLField(48, 128, lambda e: 32 * (e // 64) + e % 32),
    ),
    decode=_decode_q5_k,
)
# struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16];
# ggml_half d; }, 16 sub-blocks of 16 values with 8 bit scales. In each half
# h of 128 values, value 32 * r + l reads the nibble r // 2 of
# ql[64 * h + 32 * (r % 2) + l] and the 2 bits r of qh[32 * h + l]
Q6_K = GGMLQuantFormat(
    name="q6_K",
    qk=256,
    block_bytes=210,
    bits=6,
    fields=(
        GGMLField(0, 128, lambda e: 64 * (e // 128) + 32 * ((e // 32) % 2) + e % 32),
        GGMLField(128, 64, lambda e: 32 * (e // 128) + e % 32),
        GGMLField(192, 16, lambda e: e // 16),
        GGMLField(208),
        GGMLField(209),
    ),
    decode=_decode_q6_k,
)

GGML_QUANT_FORMATS: dict[str, GGMLQuantFormat] = {
    fmt.name: fmt for fmt in (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K)
}


def ggml_quant_format(name: str) -> GGMLQuantFormat:
    # ggml names the k-quants q4_K, accept q4_k as well
    fmt = GGML_QUANT_FORMATS.get(name)
    if fmt is None and name.endswith("_k"):
        fmt = GGML_QUANT_FORMATS.get(name[:-1] + "K")
    if fmt is None:
        raise ValueError(
            f"unsupported ggml quantization format {name!r}, "
            f"expected one of {sorted(GGML_QUANT_FORMATS)}"
        )
    return fmt


def _as_format(fmt: str | GGMLQuantFormat) -> GGMLQuantFormat:
    return ggml_quant_format(fmt) if isinstance(fmt, str) else fmt


F32Array = numpy.typing.NDArray[np.float32]
I32Array = numpy.typing.NDArray[np.int32]


class _NumpyOps:
    @staticmethod
    def f32(x: Any) -> F32Array:
        return np.asarray(x).astype(np.float32)

    @staticmethod
    def fp16(lo: Any, hi: Any) -> F32Array:
        h = np.asarray(lo | (hi << 8)).astype(np.uint16)
        return h.view("<f2").astype(np.float32)


def ggml_dequantize(
    q: numpy.typing.ArrayLike, fmt: str | GGMLQuantFormat
) -> F32Array:
    """Dequantize ggml blocks ``(..., nblocks, block_bytes)`` to ``(..., nblocks * qk)``."""
    fmt = _as_format(fmt)
    q = np.asarray(q, dtype=np.uint8)
    assert q.shape[-1] == fmt.block_bytes, (
        f"{fmt.name}: expected blocks of {fmt.block_bytes} bytes, got {q.shape[-1]}"
    )
    e = np.arange(fmt.qk, dtype=np.int32)
    fields = [
        q[..., np.broadcast_to(field.offset + field.index(e), e.shape)].astype(
            np.int32
        )
        for field in fmt.fields
    ]
    out = fmt.decode(_NumpyOps, e, *fields)
    return out.reshape(*q.shape[:-2], q.shape[-2] * fmt.qk)


#
# Quantization, ports of the ggml quantize_row_*_ref functions. Sums are
# accumulated sequentially in float32 as in C, hence the cumsums.
#

_GROUP_MAX_EPS = np.float32(1e-15)


def _f32(x: Any) -> Any:
    return np.float32(x)


def _seqsum(x: numpy.typing.NDArray[Any]) -> F32Array:
    return np.cumsum(x, axis=-1, dtype=np.float32)[..., -1]


def _nearest_int(x: F32Array) -> I32Array:
    # ggml nearest_int: round half to even
    return np.rint(x).astype(np.int32)


def _roundf(x: F32Array) -> F32Array:
    # C roundf: half away from zero (np.round is half to even)
    t = np.trunc(x)
    return t + np.where(np.abs(x - t) >= 0.5, np.sign(x), 0).astype(np.float32)


def _abs_max(x: F32Array) -> tuple[F32Array, F32Array]:
    """|x| max and the value of max magnitude, the first one on ties."""
    imax = np.argmax(np.abs(x), axis=-1)
    vmax = np.take_along_axis(x, imax[..., None], axis=-1)[..., 0]
    return np.abs(vmax), vmax


def _inv(d: F32Array) -> F32Array:
    with np.errstate(divide="ignore"):
        return np.where(d != 0, _f32(1) / d, _f32(0)).astype(np.float32)


def _fp16_bytes(d: F32Array) -> numpy.typing.NDArray[np.uint8]:
    return d.astype("<f2")[..., None].view(np.uint8)


def _fp16_round(d: F32Array) -> F32Array:
    return d.astype(np.float16).astype(np.float32)


def _quantize_q8_0(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    amax, _ = _abs_max(blocks)
    d = amax / _f32(127)
    out[..., 0:2] = _fp16_bytes(d)
    out[..., 2:] = _roundf(blocks * _inv(d)[..., None]).astype(np.int8).view(np.uint8)


def _quantize_q4_0(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    _, vmax = _abs_max(blocks)
    d = vmax / _f32(-8)
    out[..., 0:2] = _fp16_bytes(d)
    scaled = blocks * _inv(d)[..., None]
    xi = np.minimum(15, np.trunc(scaled + _f32(8.5)).astype(np.int8)).astype(np.uint8)
    out[..., 2:] = xi[..., :16] | (xi[..., 16:] << 4)


def _quantize_q5_0(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    _, vmax = _abs_max(blocks)
    d = vmax / _f32(-16)
    out[..., 0:2] = _fp16_bytes(d)
    scaled = blocks * _inv(d)[..., None]
    xi = np.minimum(31, np.trunc(scaled + _f32(16.5)).astype(np.int8)).astype(np.uint8)
    out[..., 2:6] = np.packbits(xi >> 4, axis=-1, bitorder="little")
    out[..., 6:] = (xi[..., :16] & 15) | ((xi[..., 16:] & 15) << 4)


def _make_qkx2_quants(
    nmax: int,
    x: F32Array,
    weights: F32Array,
    rmin: float,
    rdelta: float,
    nstep: int,
) -> tuple[F32Array, F32Array, I32Array]:
    """make_qkx2_quants (use_mad=False) on the groups of the last axis of x.

    Returns the scales, mins and quants of the groups.
    """
    fmin = np.minimum(x.min(axis=-1), _f32(0))
    fmax = x.max(axis=-1)
    sum_w = _seqsum(weights)
    sum_x = _seqsum(weights * x)
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = _f32(nmax) / (fmax - fmin)
        scale = _f32(1) / iscale
        L = np.clip(_nearest_int(iscale[..., None] * (x - fmin[..., None])), 0, nmax)
        diff = scale[..., None] * L.astype(np.float32) + fmin[..., None] - x
        best_error = _seqsum(weights * (diff * diff))
        for step in range(nstep + 1):
            iscale = (_f32(rmin) + _f32(rdelta) * _f32(step) + _f32(nmax)) / (
                fmax - fmin
            )
            laux = np.clip(
                _nearest_int(iscale[..., None] * (x - fmin[..., None])), 0, nmax
            )
            lf = laux.astype(np.float32)
            sum_l = _seqsum(weights * lf)
            sum_l2 = _seqsum(weights * lf * lf)
            sum_xl = _seqsum(weights * lf * x)
            D = sum_w * sum_l2 - sum_l * sum_l
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
            positive_min = this_min > 0
            this_min = np.where(positive_min, _f32(0), this_min)
            this_scale = np.where(positive_min, sum_xl / sum_l2, this_scale)
            diff = this_scale[..., None] * lf + this_min[..., None] - x
            cur_error = _seqsum(weights * (diff * diff))
            better = (D > 0) & (cur_error < best_error)
            L = np.where(better[..., None], laux, L)
            best_error = np.where(better, cur_error, best_error)
            scale = np.where(better, this_scale, scale)
            fmin = np.where(better, this_min, fmin)
    flat = fmax == fmin
    L = np.where(flat[..., None], 0, L)
    scale = np.where(flat, _f32(0), scale)
    return scale.astype(np.float32), (-fmin).astype(np.float32), L


def _quantize_k4(
    blocks: F32Array,
    out: numpy.typing.NDArray[np.uint8],
    nmax: int,
    rmin: float,
    nstep: int,
) -> I32Array:
    """Scales of q4_K/q5_K blocks, returns the quants in [0, nmax]."""
    x = blocks.reshape(*blocks.shape[:-1], 8, 32)
    av_x = np.sqrt(_seqsum(x * x) / _f32(32))
    weights = av_x[..., None] + np.abs(x)
    scales, mins, L = _make_qkx2_quants(nmax, x, weights, rmin, 0.1, nstep)
    max_scale = np.maximum(scales.max(axis=-1), _f32(0))
    max_min = np.maximum(mins.max(axis=-1), _f32(0))
    with np.errstate(divide="ignore"):
        inv_scale = np.where(max_scale > 0, _f32(63) / max_scale, _f32(0))
        inv_min = np.where(max_min > 0, _f32(63) / max_min, _f32(0))
    ls = np.minimum(63, _nearest_int(inv_scale[..., None] * scales).astype(np.uint8))
    lm = np.minimum(63, _nearest_int(inv_min[..., None] * mins).astype(np.uint8))
    sc = out[..., 4:16]
    sc[..., 0:4] = ls[..., :4] | ((ls[..., 4:] >> 4) << 6)
    sc[..., 4:8] = lm[..., :4] | ((lm[..., 4:] >> 4) << 6)
    sc[..., 8:12] = (ls[..., 4:] & 15) | ((lm[..., 4:] & 15) << 4)
    d = max_scale / _f32(63)
    dmin = max_min / _f32(63)
    out[..., 0:2] = _fp16_bytes(d)
    out[..., 2:4] = _fp16_bytes(dmin)
    dj = _fp16_round(d)[..., None] * ls.astype(np.float32)
    dm = _fp16_round(dmin)[..., None] * lm.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.clip(_nearest_int((x + dm[..., None]) / dj[..., None]), 0, nmax)
    L = np.where((dj != 0)[..., None], lq, L)
    return L.reshape(blocks.shape)


def _quantize_q4_k(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    L = _quantize_k4(blocks, out, nmax=15, rmin=-1.0, nstep=20).astype(np.uint8)
    L = L.reshape(*L.shape[:-1], 4, 2, 32)
    qs = L[..., 0, :] | (L[..., 1, :] << 4)
    out[..., 16:] = qs.reshape(*qs.shape[:-2], 128)


def _quantize_q5_k(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    L = _quantize_k4(blocks, out, nmax=31, rmin=-0.5, nstep=15).astype(np.uint8)
    L = L.reshape(*L.shape[:-1], 8, 32)
    high = (L >> 4) << np.arange(8, dtype=np.uint8)[:, None]
    out[..., 16:48] = np.bitwise_or.reduce(high, axis=-2)
    low = (L & 15).reshape(*L.shape[:-2], 4, 2, 32)
    qs = low[..., 0, :] | (low[..., 1, :] << 4)
    out[..., 48:] = qs.reshape(*qs.shape[:-2], 128)


def _make_qx_quants_rmse1(nmax: int, x: F32Array) -> tuple[F32Array, I32Array]:
    """make_qx_quants (rmse_type=1, no weights) on the groups of the last axis.

    Returns the scales and the quants in [0, 2 * nmax).
    """
    amax, vmax = _abs_max(x)
    w = x * x

    def quants(iscale: F32Array) -> tuple[I32Array, F32Array, F32Array]:
        lq = np.clip(_nearest_int(iscale[..., None] * x), -nmax, nmax - 1)
        lf = lq.astype(np.float32)
        return lq, _seqsum(w * x * lf), _seqsum(w * lf * lf)

    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = _f32(-nmax) / vmax
        lq, sumlx, suml2 = quants(iscale)
        L = lq + nmax
        scale = np.where(suml2 != 0, sumlx / suml2, _f32(0))
        best = scale * sumlx
        for step in range(-9, 10):
            if step == 0:
                continue
            iscale = -(_f32(nmax) + _f32(0.1) * _f32(step)) / vmax
            lq, sumlx, suml2 = quants(iscale)
            better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
            L = np.where(better[..., None], lq + nmax, L)
            scale = np.where(better, sumlx / suml2, scale)
            best = np.where(better, scale * sumlx, best)
    zero = amax < _GROUP_MAX_EPS
    L = np.where(zero[..., None], 0, L)
    scale = np.where(zero, _f32(0), scale)
    return scale.astype(np.float32), L


def _quantize_q6_k(blocks: F32Array, out: numpy.typing.NDArray[np.uint8]) -> None:
    x = blocks.reshape(*blocks.shape[:-1], 16, 16)
    scales, L = _make_qx_quants_rmse1(32, x)
    max_abs_scale, max_scale = _abs_max(scales)
    zero = max_abs_scale < _GROUP_MAX_EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = _f32(-128) / max_scale
        d = _f32(1) / iscale
        sc = np.minimum(127, _nearest_int(iscale[..., None] * scales))
        dj = _fp16_round(d)[..., None] * sc.astype(np.float32)
        lq = np.clip(_nearest_int(x / dj[..., None]), -32, 31) + 32
    L = np.where((dj != 0)[..., None], lq, L).astype(np.uint8)
    L = L.reshape(*blocks.shape[:-1], 2, 4, 32)
    ql = (L[..., 0:2, :] & 15) | ((L[..., 2:4, :] & 15) << 4)
    qh = np.bitwise_or.reduce(
        (L >> 4) << (2 * np.arange(4, dtype=np.uint8))[:, None], axis=-2
    )
    out[..., 0:128] = ql.reshape(*ql.shape[:-3], 128)
    out[..., 128:192] = qh.reshape(*qh.shape[:-2], 64)
    out[..., 192:208] = sc.astype(np.int8).view(np.uint8)
    out[..., 208:210] = _fp16_bytes(d)
    out[zero] = 0


_QUANTIZERS: dict[str, Callable[[F32Array, numpy.typing.NDArray[np.uint8]], None]] = {
    "q8_0": _quantize_q8_0,
    "q4_0": _quantize_q4_0,
    "q5_0": _quantize_q5_0,
    "q4_K": _quantize_q4_k,
    "q5_K": _quantize_q5_k,
    "q6_K": _quantize_q6_k,
}


def ggml_quantize(
    x: numpy.typing.ArrayLike, fmt: str | GGMLQuantFormat
) -> numpy.typing.NDArray[np.uint8]:
    """Quantize the last dimension of ``x`` in ggml blocks.

    Returns a ``uint8`` array of shape ``(*x.shape[:-1], K // qk, block_bytes)``.
    """
    fmt = _as_format(fmt)
    x = np.asarray(x, dtype=np.float32)
    k = x.shape[-1]
    assert k % fmt.qk == 0, (
        f"{fmt.name}: row size {k} is not a multiple of the block size {fmt.qk}"
    )
    nb = k // fmt.qk
    blocks = x.reshape(*x.shape[:-1], nb, fmt.qk)
    out = np.zeros((*x.shape[:-1], nb, fmt.block_bytes), dtype=np.uint8)
    _QUANTIZERS[fmt.name](blocks, out)
    return out
