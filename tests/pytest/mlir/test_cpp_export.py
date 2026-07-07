#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mlir_utils import matmul_impl, requires_mlir

I, J, K, DTYPE = 4, 16, 8, "float32"


@requires_mlir
@pytest.mark.skipif(sys.platform != "linux", reason="cpp export integration test (linux)")
def test_cpp_export_matmul(tmp_path: Path) -> None:
    impl = matmul_impl(I, J, K, DTYPE, "matmul_export")
    sch = impl.get_scheduler()
    sch.set_dims(["i", "j", "k"])
    sched = sch.schedule()

    comp = impl.get_compiler(ar_lib=True, dump_file=str(tmp_path / "matmul_export"))
    module = comp.compile(sched)

    export_dir = tmp_path / "export"
    module.export(export_dir)

    export_name = "matmul_export"
    assert (export_dir / "include" / f"{export_name}.h").is_file()
    assert (export_dir / "lib" / f"lib{export_name}.a").is_file()
    assert (export_dir / "test.cpp").is_file()
    assert (export_dir / "Makefile").is_file()
    makefile = (export_dir / "Makefile").read_text(encoding="utf-8")
    assert "test-static:" in makefile
    assert "STATIC_LDFLAGS" in makefile
    assert "-pthread -static" in makefile
    assert (export_dir / "README.md").is_file()
    assert list((export_dir / "data" / "inputs").glob("*.bin"))
    assert list((export_dir / "data" / "outputs").glob("*.bin"))

    build = subprocess.run(
        ["make", "test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, f"make failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run(
        ["./test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"test binary failed:\n{run.stdout}\n{run.stderr}"
    assert "All checks passed." in run.stdout


@requires_mlir
@pytest.mark.skipif(sys.platform != "linux", reason="cpp export integration test (linux)")
def test_cpp_export_matmul_runtime_validate(tmp_path: Path) -> None:
    impl = matmul_impl(I, J, K, DTYPE, "matmul_export_rt")
    sch = impl.get_scheduler()
    sch.set_dims(["i", "j", "k"])
    sched = sch.schedule()

    comp = impl.get_compiler(ar_lib=True, dump_file=str(tmp_path / "matmul_export_rt"))
    module = comp.compile(sched)

    export_dir = tmp_path / "export_rt"
    module.export(export_dir, runtime_validate=True, seed=7)

    assert not (export_dir / "data").exists()
    test_cpp = (export_dir / "test.cpp").read_text(encoding="utf-8")
    assert "fill_random_inputs" in test_cpp
    assert "reference_matmul" in test_cpp
    assert "load_binary" not in test_cpp

    build = subprocess.run(
        ["make", "test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, f"make failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run(
        ["./test"],
        cwd=export_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"test binary failed:\n{run.stdout}\n{run.stderr}"
    assert "All checks passed." in run.stdout


@requires_mlir
@pytest.mark.skipif(sys.platform != "linux", reason="cpp export integration test (linux)")
@pytest.mark.skipif(
    shutil.which("aarch64-linux-gnu-g++") is None
    and shutil.which("aarch64-linux-gnu-gcc") is None,
    reason="aarch64 cross compiler not installed",
)
def test_cpp_export_matmul_cross_aarch64(tmp_path: Path) -> None:
    impl = matmul_impl(I, J, K, DTYPE, "matmul_export_aarch64")
    sch = impl.get_scheduler()
    sch.set_dims(["i", "j", "k"])
    sched = sch.schedule()

    comp = impl.get_compiler(
        ar_lib=True,
        arch="aarch64",
        cpu="generic",
        dump_file=str(tmp_path / "matmul_export_aarch64"),
    )
    module = comp.compile(sched)

    export_dir = tmp_path / "export_aarch64"
    module.export(export_dir)

    export_name = "matmul_export_aarch64"
    ar_path = export_dir / "lib" / f"lib{export_name}.a"
    assert ar_path.is_file()

    makefile = (export_dir / "Makefile").read_text(encoding="utf-8")
    assert "aarch64-linux-gnu-g++" in makefile

    file_out = subprocess.run(
        ["file", str(ar_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert file_out.returncode == 0, file_out.stderr
    assert "aarch64" in file_out.stdout.lower() or "arm" in file_out.stdout.lower()
