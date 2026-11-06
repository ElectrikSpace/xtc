#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Pending kernels: stubs written by llama.cpp (ggml-xtc) for kernels missing from the base.

A stub lives in ``<kernelbase>/_pending/<key>/`` and holds:

* ``graph.yaml``      the exact subgraph (must hash to ``<key>``, do not edit),
* ``build_kernel.py`` a runnable script where the user only provides the descript schedule
  (``SPEC``). Running it compiles, validates against the reference and inserts the kernel.

(Older stubs with a ``schedule.yaml`` instead of ``build_kernel.py`` are still supported.)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from xtc.graphs.xtc.builder import graph_builder

from .canonical import graph_key
from .schedule import DescriptSchedule
from .store import KernelBase

__all__ = [
    "PENDING_DIR",
    "pending_root",
    "pending_keys",
    "stub_script",
    "stub_is_todo",
    "build_stub",
    "run_stub",
]

PENDING_DIR = "_pending"
SCRIPT_NAME = "build_kernel.py"

_TODO_RE = re.compile(r"^TODO\s*=\s*True\b", re.MULTILINE)


def pending_root(kb: KernelBase) -> Path:
    return Path(kb.root) / PENDING_DIR


def stub_script(stub_dir: Path) -> Path:
    return stub_dir / SCRIPT_NAME


def _is_stub(d: Path) -> bool:
    return d.is_dir() and (d / "graph.yaml").is_file() and (
        stub_script(d).is_file() or (d / "schedule.yaml").is_file()
    )


def pending_keys(kb: KernelBase) -> list[str]:
    root = pending_root(kb)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if _is_stub(d))


def stub_is_todo(stub_dir: Path) -> bool:
    """True while the user has not provided the schedule yet."""
    script = stub_script(stub_dir)
    if script.is_file():
        return bool(_TODO_RE.search(script.read_text(encoding="utf-8")))
    with open(stub_dir / "schedule.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return bool(isinstance(data, dict) and data.get("todo", False))


def build_stub(
    kb: KernelBase,
    key: str,
    sched: DescriptSchedule,
    overwrite: bool = False,
) -> bool:
    """Compile/validate/insert the pending kernel ``key`` with ``sched``. Keeps the stub on failure."""
    d = pending_root(kb) / key
    try:
        graph = graph_builder.graph_from_file(d / "graph.yaml")
        got = graph_key(graph)
        if got != key:
            print(
                f"{key}: graph.yaml hashes to {got}, not {key}. The graph must not be "
                "modified, otherwise llama.cpp would never find the kernel.",
                file=sys.stderr,
            )
            return False
        entry = kb.add(graph, sched, overwrite=overwrite)
    except Exception as e:  # compile / schedule errors: keep the stub for the user
        traceback.print_exc()
        print(f"{key}: build failed: {e}", file=sys.stderr)
        return False
    print(f"{key}: compiled, running {entry.test_py} ...", flush=True)
    r = subprocess.run([sys.executable, str(entry.test_py)], check=False)
    if r.returncode != 0:
        print(
            f"{key}: the kernel does NOT validate against the reference; removed from "
            "the kernel base, stub kept. Fix the schedule and rebuild.",
            file=sys.stderr,
        )
        kb.remove(key)
        return False
    shutil.rmtree(d)
    print(f"{key}: OK, stored in {entry.path}")
    return True


def run_stub(
    script_file: str | Path,
    node_name: str,
    abstract_dims: list[str],
    spec: dict[str, Any] | str,
    todo: bool = False,
    abstract_dim_sizes: dict[str, int] | None = None,
    abstract_matrix: list[str] | None = None,
    sample: dict[str, int] | None = None,
    default_node: str | None = None,
    argv: list[str] | None = None,
) -> int:
    """Entry point of a stub's ``build_kernel.py``; returns a process exit code."""
    script = Path(script_file).resolve()
    stub_dir = script.parent
    key = stub_dir.name

    ap = argparse.ArgumentParser(description=f"Build the pending XTC kernel {key}")
    ap.add_argument("--root", type=Path, default=None, help="kernel base root (default: inferred)")
    ap.add_argument("--force", action="store_true", help="build even if TODO = True")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing kernel")
    ns = ap.parse_args(argv)

    # <root>/_pending/<key>/build_kernel.py
    root = ns.root or (stub_dir.parent.parent if stub_dir.parent.name == PENDING_DIR else None)
    kb = KernelBase(root)
    if todo and not ns.force:
        print(
            f"{key}: schedule not provided yet. Fill SPEC in {script} and set TODO = False "
            "(or use --force to build the baseline).",
            file=sys.stderr,
        )
        return 2
    sched = DescriptSchedule(
        node_name=node_name,
        abstract_dims=list(abstract_dims),
        spec=spec,
        abstract_dim_sizes=dict(abstract_dim_sizes or {}),
        abstract_matrix=list(abstract_matrix or []),
        sample=dict(sample or {}),
        default_node=default_node,
    )
    return 0 if build_stub(kb, key, sched, overwrite=ns.overwrite) else 1
