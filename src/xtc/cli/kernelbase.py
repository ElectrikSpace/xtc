#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from xtc.graphs.xtc.builder import graph_builder
from xtc.kernelbase import DescriptSchedule, KernelBase, graph_hash, graph_key, pending


def cmd_add(args: argparse.Namespace) -> int:
    kb = KernelBase(args.root)
    graph = graph_builder.graph_from_file(args.graph)
    sched = DescriptSchedule.load(args.schedule)
    bk_kw: dict = {}
    if args.backend_kwargs and str(args.backend_kwargs).strip():
        loaded = yaml.safe_load(args.backend_kwargs)
        assert isinstance(loaded, dict)
        bk_kw = loaded
    entry = kb.add(
        graph,
        sched,
        backend=args.backend,
        backend_kwargs=bk_kw,
        overwrite=args.overwrite,
    )
    print(f"Added kernel {entry.key} at {entry.path}")
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    kb = KernelBase(args.root)
    graph = graph_builder.graph_from_file(args.graph)
    key = graph_key(graph)
    h = graph_hash(graph)
    entry = kb.lookup(graph)
    print(f"key={key}")
    print(f"sha256={h}")
    if entry is None:
        print("not found")
        return 1
    print(f"found: {entry.path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    kb = KernelBase(args.root)
    for k in kb.keys():
        print(k)
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    kb = KernelBase(args.root)
    keys = args.keys if args.keys else kb.keys()
    if not keys:
        print("no entries", file=sys.stderr)
        return 0
    failed = 0
    for k in keys:
        entry = kb.get(k)
        if entry is None or not entry.test_py.is_file():
            print(f"missing entry or test: {k}", file=sys.stderr)
            failed += 1
            continue
        print(f"Running {entry.test_py} ...", flush=True)
        r = subprocess.run([sys.executable, str(entry.test_py)], check=False)
        if r.returncode != 0:
            failed += 1
    return 1 if failed else 0


# stubs written by llama.cpp for missing kernels, see xtc.kernelbase.pending
_pending_root = pending.pending_root
_pending_keys = pending.pending_keys
_stub_is_todo = pending.stub_is_todo


def cmd_pending(args: argparse.Namespace) -> int:
    """List the stubs written by llama.cpp for missing kernels."""
    kb = KernelBase(args.root)
    keys = _pending_keys(kb)
    if not keys:
        print("no pending kernels")
        return 0
    for k in keys:
        d = _pending_root(kb) / k
        state = "TODO" if _stub_is_todo(d) else "ready"
        g = yaml.safe_load(open(d / "graph.yaml", encoding="utf-8"))
        shapes = " x ".join(str(tuple(i["expr"]["type"]["shape"])) for i in g["inputs"])
        todo_file = pending.stub_script(d)
        if not todo_file.is_file():
            todo_file = d / "schedule.yaml"
        print(f"{k}  [{state}]  {g.get('name', '?')}  {shapes}  {todo_file}")
    return 0


def _build_one(kb: KernelBase, key: str, force: bool, overwrite: bool) -> bool:
    d = _pending_root(kb) / key
    script = pending.stub_script(d)
    if not (d / "graph.yaml").is_file() or not (script.is_file() or (d / "schedule.yaml").is_file()):
        print(f"{key}: no such pending stub in {_pending_root(kb)}", file=sys.stderr)
        return False
    if script.is_file():
        # the stub script is self-contained: it holds the schedule and does the build
        cmd = [sys.executable, str(script), "--root", str(kb.root)]
        if force:
            cmd.append("--force")
        if overwrite:
            cmd.append("--overwrite")
        return subprocess.run(cmd, check=False).returncode == 0
    # legacy stub: schedule.yaml
    if not force and _stub_is_todo(d):
        print(
            f"{key}: schedule not implemented yet. Edit {d / 'schedule.yaml'} "
            "and delete the `todo: true` line (or use --force).",
            file=sys.stderr,
        )
        return False
    try:
        sched = DescriptSchedule.load(d / "schedule.yaml")
    except Exception as e:
        print(f"{key}: invalid schedule.yaml: {e}", file=sys.stderr)
        return False
    return pending.build_stub(kb, key, sched, overwrite=overwrite)


def cmd_build(args: argparse.Namespace) -> int:
    """Compile pending stubs whose schedule is done, validate them, insert in the base."""
    kb = KernelBase(args.root)
    keys = args.keys or _pending_keys(kb)
    if not keys:
        print("no pending kernels")
        return 0
    failed = 0
    for k in keys:
        if not args.keys and _stub_is_todo(_pending_root(kb) / k) and not args.force:
            print(f"{k}: skipped (schedule not provided yet, see `xtc-kernelbase pending`)")
            failed += 1
            continue
        if not _build_one(kb, k, args.force, args.overwrite):
            failed += 1
    if failed:
        print(f"{failed} kernel(s) not built", file=sys.stderr)
    return 1 if failed else 0


def _add_root_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="kernel base root (default: $XTC_KERNELBASE or ./xtc_kernelbase)",
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="xtc-kernelbase", description="XTC kernel base tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add", help="compile and store a kernel")
    _add_root_arg(pa)
    pa.add_argument("--graph", type=Path, required=True)
    pa.add_argument("--schedule", type=Path, required=True)
    pa.add_argument("--backend", default="mlir")
    pa.add_argument(
        "--backend-kwargs",
        type=str,
        default="",
        help="YAML dict for Backend(), e.g. '{use_tensor_dialect: true}'",
    )
    pa.add_argument("--overwrite", action="store_true")
    pa.set_defaults(func=cmd_add)

    pl = sub.add_parser("lookup", help="print key/hash for a graph yaml")
    _add_root_arg(pl)
    pl.add_argument("--graph", type=Path, required=True)
    pl.set_defaults(func=cmd_lookup)

    pli = sub.add_parser("list", help="list stored kernel keys")
    _add_root_arg(pli)
    pli.set_defaults(func=cmd_list)

    pt = sub.add_parser("test", help="run stored test_kernel.py")
    _add_root_arg(pt)
    pt.add_argument("keys", nargs="*", help="kernel keys (default: all)")
    pt.set_defaults(func=cmd_test)

    pp = sub.add_parser("pending", help="list stubs written by llama.cpp for missing kernels")
    _add_root_arg(pp)
    pp.set_defaults(func=cmd_pending)

    pb = sub.add_parser(
        "build", help="compile+validate pending stubs (todo removed) and insert them"
    )
    _add_root_arg(pb)
    pb.add_argument("keys", nargs="*", help="pending keys (default: all)")
    pb.add_argument("--force", action="store_true", help="build even if the schedule is still marked TODO")
    pb.add_argument("--overwrite", action="store_true")
    pb.set_defaults(func=cmd_build)

    ns = p.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
