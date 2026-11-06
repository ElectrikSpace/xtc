#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Formats of the weights of llama.cpp quantized models.

A llama.cpp model quantization type ("ftype", e.g. ``Q4_K_M``) is not a
single ggml format but a recipe choosing a format per tensor
(``llama_tensor_get_type`` in ``src/llama-quant.cpp``): the most sensitive
tensors get more bits. 1-D tensors (norms, biases) stay in F32.
"""

from __future__ import annotations

import re

from .ggml import ggml_quant_format

__all__ = [
    "q4_k_m_tensor_format",
]

# ncols fallbacks of the 256 values blocks formats (tensor_type_fallback)
_FALLBACKS = {"q4_K": "q5_0", "q5_K": "q5_1", "q6_K": "q8_0"}
_BLOCK_SIZES = {"q5_1": 32}


def _use_more_bits(i_layer: int, n_layers: int) -> bool:
    return (
        i_layer < n_layers // 8
        or i_layer >= 7 * n_layers // 8
        or (i_layer - n_layers // 8) % 3 == 2
    )


def _block_size(fmt: str) -> int:
    return _BLOCK_SIZES.get(fmt) or ggml_quant_format(fmt).qk


def q4_k_m_tensor_format(
    name: str, ncols: int, n_layer: int, tied_embeddings: bool = False
) -> str:
    """ggml format of the 2-D weight ``name`` of a ``Q4_K_M`` model.

    Mostly ``q4_K``, with ``q6_K`` for the output and for half of the
    ``attn_v`` and ``ffn_down`` weights (first and last eighths of the layers
    and every third layer in between), ``q5_K`` for fused ``attn_qkv``.
    Weights whose rows (``ncols`` values, the reduction dimension of the
    matmuls) are not made of whole 256 values blocks fall back to 32 values
    blocks formats (``q5_0``, ``q5_1``, ``q8_0``), else to ``f16``.

    Covers dense models, without the architecture specific rules of
    llama.cpp (Falcon, 8 experts MoE, 70B ``attn_v``).

    Args:
        name: GGUF tensor name, e.g. ``"blk.3.ffn_down.weight"``.
        ncols: row size of the tensor (GGUF ``ne[0]``).
        n_layer: number of layers (blocks) of the model.
        tied_embeddings: the model has no ``output.weight``, the token
            embeddings are used as output.
    """
    layer = re.match(r"blk\.(\d+)\.", name)
    i_layer = int(layer.group(1)) if layer else -1
    if name == "output.weight" or (tied_embeddings and name == "token_embd.weight"):
        return "q6_K" if ncols % 256 == 0 else "q8_0"
    if name.endswith(("attn_v.weight", "ffn_down.weight")):
        fmt = "q6_K" if _use_more_bits(i_layer, n_layer) else "q4_K"
    elif name.endswith("attn_qkv.weight"):
        fmt = "q5_K"
    else:
        fmt = "q4_K"
    if ncols % _block_size(fmt) != 0:
        fmt = _FALLBACKS[fmt]
        if ncols % _block_size(fmt) != 0:
            fmt = "f16"
    return fmt
