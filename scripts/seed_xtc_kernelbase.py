#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Populate ../xtc_kernelbase with ggml-style matmul kernels (tiny test model) and a matmul+relu example (requires MLIR)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo layout: xtc/scripts/this_file.py
_XTC_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _XTC_ROOT.parent
if str(_XTC_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_XTC_ROOT / "src"))

import xtc.graphs.xtc.op as O

from xtc.kernelbase import DescriptSchedule, KernelBase, ggml_matmul_graph


# (M, N, K) of the F32 weight matmuls of the test models, with `-ub 1` (M=1)
# tiny-llama: llama.cpp/ggml/src/ggml-xtc/scripts/make_tiny_llama.py
TINY_LLAMA_MATMULS = [(1, 64, 64), (1, 32, 64), (1, 128, 64), (1, 64, 128), (1, 259, 64)]
# stories260K: llama.cpp/testons/smoke_test.sh (ggml-org/models tinyllamas/stories260K.gguf)
STORIES260K_MATMULS = [(1, 64, 64), (1, 32, 64), (1, 172, 64), (1, 64, 172), (1, 512, 64)]


def seed_ggml_matmuls(kb: KernelBase, overwrite: bool) -> None:
    """ggml MUL_MAT kernels, the exact subgraphs ggml-xtc looks up from llama.cpp."""
    for m, n, k in sorted(set(TINY_LLAMA_MATMULS + STORIES260K_MATMULS)):
        graph = ggml_matmul_graph(m, n, k)
        if not overwrite and kb.lookup(graph) is not None:
            print(f"ggml matmul M={m} N={n} K={k}: already in the kernel base, skipped")
            continue
        sched = DescriptSchedule(
            node_name="C",
            abstract_dims=["i", "j", "k"],
            spec={
                "k": {},
                "i": {},
                "j": {},
                f"k#{8 if k % 8 == 0 else 4}": {"unroll": True},
                "j#8": {"vectorize": True},
            },
        )
        e = kb.add(graph, sched, overwrite=overwrite)
        print(f"ggml matmul M={m} N={n} K={k}: {e.key} -> {e.path}")


def seed_matmul_relu(kb: KernelBase, overwrite: bool) -> None:
    I, J, K, dtype = 4, 32, 512, "float32"
    a = O.tensor((I, K), dtype, name="A")
    b = O.tensor((K, J), dtype, name="B")
    with O.graph(name="matmul_relu") as gb:
        m = O.matmul(a, b, name="matmul")
        O.relu(m, name="relu")
    graph = gb.graph
    if not overwrite and kb.lookup(graph) is not None:
        print("matmul_relu: already in the kernel base, skipped")
        return
    sched = DescriptSchedule(
        node_name="matmul",
        abstract_dims=["i", "j", "k"],
        spec={
            "i": {},
            "j": {},
            "i#2": {},
            "j#16": {"fuse_consumer": True},
            "k": {},
        },
        default_node="matmul",
    )
    e = kb.add(
        graph,
        sched,
        backend_kwargs={"use_tensor_dialect": True},
        overwrite=overwrite,
    )
    print(f"matmul_relu: {e.key} -> {e.path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=_WORKSPACE_ROOT / "xtc_kernelbase",
        help="kernel base directory",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    kb = KernelBase(args.root)
    seed_ggml_matmuls(kb, args.overwrite)
    seed_matmul_relu(kb, args.overwrite)


if __name__ == "__main__":
    main()
