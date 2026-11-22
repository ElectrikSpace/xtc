#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import hashlib
import json
from typing import Any

from xtc.graphs.xtc.graph import XTCGraph

__all__ = [
    "canonical_graph_dict",
    "canonical_graph_json",
    "graph_hash",
    "graph_key",
]


def canonical_graph_dict(graph: XTCGraph, target: str | None = None) -> dict[str, Any]:
    """Graph representation used for kernel base keys.

    Kept: operators (name and attributes), graph structure, tensor shapes,
    dtypes, layouts and const flags.
    Dropped: graph name, node/tensor display names and ops_version.
    Unique ids are renumbered ``%0, %1, ...`` in order of appearance (inputs,
    then nodes), so the key does not depend on global expression counters.
    ``target`` (e.g. "mppa") is added when set; host kernels have none.

    The same canonical form is computed natively by the llama.cpp ggml-xtc
    backends for MUL_MAT nodes, so any change here must be mirrored there.
    """
    d = graph.to_dict()
    uid_map: dict[str, str] = {}
    for item in [*d["inputs"], *d["nodes"]]:
        uid_map[item["uid"]] = f"%{len(uid_map)}"

    def conv(item: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in item.items() if k != "name"}
        out["uid"] = uid_map[item["uid"]]
        expr = out.get("expr")
        if isinstance(expr, dict) and "args" in expr:
            out["expr"] = {**expr, "args": [uid_map[a] for a in expr["args"]]}
        return out

    out: dict[str, Any] = {
        "inputs": [conv(i) for i in d["inputs"]],
        "nodes": [conv(n) for n in d["nodes"]],
        "outputs": [{"uid": uid_map[o["uid"]]} for o in d["outputs"]],
    }
    if target is not None:
        out["target"] = target
    return out


def canonical_graph_json(graph: XTCGraph, target: str | None = None) -> str:
    return json.dumps(
        canonical_graph_dict(graph, target),
        sort_keys=True,
        separators=(",", ":"),
    )


def graph_hash(graph: XTCGraph, target: str | None = None) -> str:
    """Full sha256 (hex) of the canonical graph JSON."""
    return hashlib.sha256(canonical_graph_json(graph, target).encode("utf-8")).hexdigest()


def graph_key(graph: XTCGraph, prefix_len: int = 16, target: str | None = None) -> str:
    return graph_hash(graph, target)[:prefix_len]
