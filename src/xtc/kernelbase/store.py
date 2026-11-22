#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import importlib
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from xtc.graphs.xtc.graph import XTCGraph
from xtc.graphs.xtc.operators import XTCOperator

from .canonical import canonical_graph_dict, graph_hash, graph_key
from .schedule import DescriptSchedule
from .test_template import render_test_kernel_py

__all__ = [
    "KernelBase",
    "KernelEntry",
]


def default_kernelbase_root() -> Path:
    env = os.environ.get("XTC_KERNELBASE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / "xtc_kernelbase"


@dataclass
class KernelEntry:
    root: Path
    key: str

    @property
    def path(self) -> Path:
        return self.root / self.key

    @property
    def kernel_so(self) -> Path:
        return self.path / "kernel.so"

    @property
    def kernel_obj(self) -> Path:
        return self.path / "kernel.o"

    @property
    def static_lib(self) -> Path:
        return self.path / f"libxtc_k_{self.key}.a"

    @property
    def graph_yaml(self) -> Path:
        return self.path / "graph.yaml"

    @property
    def schedule_yaml(self) -> Path:
        return self.path / "schedule.yaml"

    @property
    def meta_yaml(self) -> Path:
        return self.path / "meta.yaml"

    @property
    def test_py(self) -> Path:
        return self.path / "test_kernel.py"

    def load_meta(self) -> dict[str, Any]:
        with open(self.meta_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        return data


class KernelBase:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else default_kernelbase_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve_backend(self, backend: str):
        mod = importlib.import_module(f"xtc.backends.{backend}")
        backend_cls = getattr(mod, "Backend", None)
        if backend_cls is None:
            raise ValueError(f"xtc.backends.{backend} has no Backend class")
        return backend_cls

    def add(
        self,
        graph: XTCGraph,
        schedule: DescriptSchedule,
        *,
        backend: str = "mlir",
        backend_kwargs: dict[str, Any] | None = None,
        overwrite: bool = False,
        bare_ptr: bool = True,
        target: str | None = None,
        validate: bool = True,
    ) -> KernelEntry:
        if target not in (None, "mppa"):
            raise ValueError(f"unknown kernel target {target!r}")
        backend_kwargs = dict(backend_kwargs or {})
        key = graph_key(graph, target=target)
        full_hash = graph_hash(graph, target=target)
        entry = KernelEntry(self.root, key)
        if entry.path.exists() and not overwrite:
            raise FileExistsError(
                f"kernel base entry {key} already exists (use overwrite=True)"
            )

        entry.path.mkdir(parents=True, exist_ok=True)
        graph.dump(entry.graph_yaml)

        sch_dict = schedule.to_dict()
        with open(entry.schedule_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(sch_dict, f, sort_keys=False)

        Backend = self._resolve_backend(backend)
        impl = Backend(graph, **backend_kwargs)
        sched_kw: dict[str, Any] = {}
        if schedule.default_node is not None:
            sched_kw["default_node"] = schedule.default_node
        sch_impl = impl.get_scheduler(**sched_kw)
        if target == "mppa":
            from xtc.runtimes.accelerator.mppa import MppaDevice

            nb_pe = MppaDevice().nb_pe
            sch_impl.define_memory_mesh(axes={"mx": 1})
            sch_impl.define_processor_mesh(axes={"px": 1, "psx": nb_pe})
        schedule.apply(sch_impl)
        mlir_sched = sch_impl.schedule()

        if target == "mppa":
            module = self._build_mppa(impl, mlir_sched, bare_ptr, entry, validate)
        else:
            with tempfile.TemporaryDirectory(prefix="xtc_kb_build_") as tmpd:
                tmp_dump = Path(tmpd) / "kernel"
                comp = impl.get_compiler(
                    shared_lib=True,
                    dump_file=str(tmp_dump),
                    bare_ptr=bare_ptr,
                    save_temps=True,
                    save_temps_dir=tmpd,
                )
                module = comp.compile(schedule=mlir_sched)

                src_so = Path(module.file_name)
                if not src_so.is_file():
                    raise FileNotFoundError(f"compiled shared library missing: {src_so}")
                shutil.copy2(src_so, entry.kernel_so)

                src_obj = Path(tmpd) / "kernel.o"
                if not src_obj.is_file():
                    raise FileNotFoundError(f"compiled object missing: {src_obj}")
                shutil.copy2(src_obj, entry.kernel_obj)

        canonical = canonical_graph_dict(graph, target)
        try:
            from importlib.metadata import version as pkg_version

            xtc_ver = pkg_version("xtc-tools")
        except Exception:
            xtc_ver = "unknown"

        meta: dict[str, Any] = {
            "key": key,
            "sha256": full_hash,
            "canonical_graph": canonical,
            "payload_name": module.payload_name,
            "object": "kernel.o" if target is None else None,
            "graph_name": graph.name,
            "backend": backend,
            "backend_kwargs": backend_kwargs,
            "bare_ptr": bare_ptr,
            "xtc_version": xtc_ver,
            "ops_version": graph.to_dict().get("ops_version", XTCOperator.version_string()),
            "target_machine": platform.machine(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        if target is not None:
            meta["target"] = target
        with open(entry.meta_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

        with open(entry.test_py, "w", encoding="utf-8") as f:
            f.write(render_test_kernel_py(entry.path, target))

        return entry

    def _build_mppa(
        self, impl: Any, sched: Any, bare_ptr: bool, entry: KernelEntry, validate: bool = True
    ) -> Any:
        from xtc.runtimes.accelerator.mppa import MppaDevice

        # the host runner stores the path of the device kernel: build it in the entry directory
        comp = impl.get_compiler(
            target=MppaDevice(),
            shared_lib=True,
            dump_file=str(entry.path / "kernel"),
            bare_ptr=bare_ptr,
        )
        module = comp.compile(schedule=sched)
        if validate and module.get_executor(validate=True).execute() != 0:
            shutil.rmtree(entry.path)
            raise RuntimeError("MPPA kernel does not validate against the reference")
        return module

    def lookup(self, graph: XTCGraph, target: str | None = None) -> KernelEntry | None:
        key = graph_key(graph, target=target)
        full_hash = graph_hash(graph, target=target)
        cand = KernelEntry(self.root, key)
        if not cand.path.is_dir():
            return None
        try:
            meta = cand.load_meta()
        except OSError:
            return None
        if meta.get("sha256") != full_hash:
            return None
        return cand

    def get(self, key: str) -> KernelEntry | None:
        e = KernelEntry(self.root, key)
        return e if e.path.is_dir() else None

    def keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "meta.yaml").is_file()
        )

    def entries(self) -> Iterator[KernelEntry]:
        for k in self.keys():
            yield KernelEntry(self.root, k)

    def remove(self, key: str) -> None:
        e = KernelEntry(self.root, key)
        if e.path.is_dir():
            shutil.rmtree(e.path)
