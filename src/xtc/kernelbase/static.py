#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Static libraries for kernel base entries, and the bundle that links them.

Each entry gets libxtc_k_<key>.a: the kernel object with its entry point renamed to xtc_raw_<key>
(all other symbols local), and wrapper.c, which exports xtc_k_<key>. The wrapper checks the ggml
tensors (type, shape, packed layout) and then calls the kernel on their data.
The bundle libxtc_kernels.so links all the archives and exports only the registry (xtc_kernel_abi.h).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from xtc.graphs.xtc.builder import graph_builder

from .schedule import DescriptSchedule
from .store import KernelBase, KernelEntry

__all__ = [
    "build_bundle",
    "build_static",
    "ensure_static",
    "render_registry",
    "render_wrapper",
]

_GGML_TYPE = {"float32": ("GGML_TYPE_F32", 4), "float16": ("GGML_TYPE_F16", 2)}

_EXPORTS_MAP = """{
    global: xtc_kernel_abi_version; xtc_kernel_registry; xtc_kernel_registry_size;
    local: *;
};
"""


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr}")


def _ggml_ne(shape: list[int], transposed: bool) -> tuple[int, int]:
    """ggml (ne0, ne1) of a packed 1D or 2D XTC tensor (ne0 is the contiguous dimension)."""
    if len(shape) == 1:
        return int(shape[0]), 1
    d0, d1 = (int(x) for x in shape)
    return (d0, d1) if transposed else (d1, d0)


def render_wrapper(key: str, graph_yaml: Path) -> str:
    """C wrapper xtc_k_<key>(srcs, dst) for the kernel described by graph_yaml."""
    doc = yaml.safe_load(graph_yaml.read_text(encoding="utf-8"))
    inputs: list[dict[str, Any]] = doc["inputs"]
    out = graph_builder.graph_from_file(graph_yaml).outputs_types[0]
    out_shape = [int(x) for x in out.shape]
    out_dtype = str(out.dtype)

    checks = []
    for i, inp in enumerate(inputs):
        t = inp["expr"]["type"]
        ne0, ne1 = _ggml_ne(t["shape"], t.get("layout") == [1, 0])
        gtype, elt = _GGML_TYPE[t["dtype"]]
        checks.append(f"    if (xtc_kernel_check(srcs[{i}], {gtype}, {elt}, {ne0}, {ne1})) {{\n        return 1;\n    }}")
    ne0, ne1 = _ggml_ne(out_shape, False)
    gtype, elt = _GGML_TYPE[out_dtype]
    checks.append(f"    if (xtc_kernel_check(dst, {gtype}, {elt}, {ne0}, {ne1})) {{\n        return 1;\n    }}")

    n = len(inputs)
    args = ", ".join([f"srcs[{i}]->data" for i in range(n)] + ["dst->data"])
    params = ", ".join(["void *"] * (n + 1))
    return (
        '#include "ggml.h"\n'
        '#include "xtc_kernel_wrap.h"\n'
        "\n"
        f"void xtc_raw_{key}({params});\n"
        "\n"
        f"int xtc_k_{key}(const struct ggml_tensor * const * srcs, struct ggml_tensor * dst) {{\n"
        + "\n".join(checks)
        + "\n"
        f"    xtc_raw_{key}({args});\n"
        "    return 0;\n"
        "}\n"
    )


def render_registry(entries: list[KernelEntry]) -> str:
    """C registry of the bundle (xtc_kernel_abi.h)."""
    lines = ['#include "xtc_kernel_abi.h"', ""]
    rows = []
    for e in entries:
        meta = e.load_meta()
        n = len(meta["canonical_graph"]["inputs"])
        lines.append(f"int xtc_k_{e.key}(const struct ggml_tensor * const * srcs, struct ggml_tensor * dst);")
        rows.append(f'    {{ "{e.key}", {n}, xtc_k_{e.key} }},')
    lines += [
        "",
        "const uint32_t xtc_kernel_abi_version = XTC_KERNEL_ABI_VERSION;",
        "const xtc_kernel_entry xtc_kernel_registry[] = {",
        *rows,
        "};",
        f"const size_t xtc_kernel_registry_size = {len(entries)};",
        "",
    ]
    return "\n".join(lines)


def build_static(entry: KernelEntry, ggml_include: Path, support_include: Path, cc: str = "cc") -> Path:
    """Write entry.static_lib from entry.kernel_obj and the generated wrapper.c."""
    if not entry.kernel_obj.is_file():
        raise FileNotFoundError(f"{entry.kernel_obj} missing (rebuild the entry)")
    meta = entry.load_meta()
    payload = meta["payload_name"]
    raw = f"xtc_raw_{entry.key}"

    wrapper_c = entry.path / "wrapper.c"
    wrapper_c.write_text(render_wrapper(entry.key, entry.graph_yaml), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="xtc_static_") as tmp:
        local_obj = Path(tmp) / "kernel.o"
        _run(
            [
                "objcopy",
                f"--redefine-sym={payload}={raw}",
                f"--keep-global-symbol={raw}",
                str(entry.kernel_obj),
                str(local_obj),
            ]
        )
        wrapper_o = Path(tmp) / "wrapper.o"
        _run(
            [
                cc, "-O2", "-fPIC", "-c", str(wrapper_c), "-o", str(wrapper_o),
                f"-I{ggml_include}", f"-I{support_include}",
            ]
        )
        entry.static_lib.unlink(missing_ok=True)
        _run(["ar", "rcs", str(entry.static_lib), str(local_obj), str(wrapper_o)])
    return entry.static_lib


def ensure_static(
    kb: KernelBase,
    entry: KernelEntry,
    ggml_include: Path,
    support_include: Path,
    cc: str = "cc",
    force: bool = False,
) -> Path:
    """Build the static lib of an entry, recompiling the kernel object first if the entry has none."""
    if entry.static_lib.is_file() and not force:
        return entry.static_lib
    if force or not entry.kernel_obj.is_file():
        meta = entry.load_meta()
        kb.add(
            graph_builder.graph_from_file(entry.graph_yaml),
            DescriptSchedule.load(entry.schedule_yaml),
            backend=meta.get("backend", "mlir"),
            backend_kwargs=meta.get("backend_kwargs") or {},
            overwrite=True,
            bare_ptr=meta.get("bare_ptr", True),
        )
    return build_static(entry, ggml_include, support_include, cc)


def build_bundle(entries: list[KernelEntry], out: Path, support_include: Path, cc: str = "cc") -> Path:
    """Link the static libs of entries into a shared library exporting the registry only."""
    if not entries:
        raise ValueError("no kernel to bundle")
    missing = [e.key for e in entries if not e.static_lib.is_file()]
    if missing:
        raise FileNotFoundError(f"no static lib for: {', '.join(missing)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xtc_bundle_") as tmp:
        registry_c = Path(tmp) / "registry.c"
        registry_c.write_text(render_registry(entries), encoding="utf-8")
        registry_o = Path(tmp) / "registry.o"
        _run([cc, "-O2", "-fPIC", "-c", str(registry_c), "-o", str(registry_o), f"-I{support_include}"])
        version = Path(tmp) / "exports.map"
        version.write_text(_EXPORTS_MAP, encoding="utf-8")
        _run(
            [
                cc, "-shared", "-o", str(out), str(registry_o),
                *[str(e.static_lib) for e in entries],
                f"-Wl,--version-script={version}", "-lm",
            ]
        )
    return out
