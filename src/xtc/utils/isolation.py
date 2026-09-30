#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Process boundary for native code that can abort the interpreter."""

import multiprocessing
import os
import pickle
import resource
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from xtc.errors import XtcError

T = TypeVar("T")
_stage_file: Path | None = None


def mark_isolated_stage(stage: str) -> None:
    """Record the active native stage for failures that cannot raise."""
    if _stage_file is not None:
        _stage_file.write_text(stage)


def _worker(
    operation: Callable[[], T], log: Path, result: Path, stage_file: Path
) -> None:
    global _stage_file
    _stage_file = stage_file
    _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard_limit))
    with log.open("wb", buffering=0) as output:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        try:
            payload: tuple[str, object] = ("result", operation())
        except Exception as exc:
            if not isinstance(exc, XtcError):
                traceback.print_exc()
            try:
                pickle.dumps(exc)
            except (pickle.PickleError, TypeError, AttributeError):
                payload = ("unserializable", traceback.format_exc())
            else:
                payload = ("error", exc)
        with result.open("wb") as sink:
            pickle.dump(payload, sink)
        sys.stdout.flush()
        sys.stderr.flush()


def run_isolated(
    operation: Callable[[], T],
    *,
    error_type: type[XtcError],
    stage: str,
) -> T:
    """Run native work in a forked process; return only serializable results.

    The caller must not depend on changes to in-memory objects in the worker.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("Native crash isolation requires a fork-capable platform")
    with tempfile.TemporaryDirectory(prefix="xtc-isolation-") as directory:
        root = Path(directory)
        log, result, stage_file = (
            root / "output.log",
            root / "result.pkl",
            root / "stage",
        )
        process = multiprocessing.get_context("fork").Process(
            target=_worker, args=(operation, log, result, stage_file)
        )
        sys.stdout.flush()
        sys.stderr.flush()
        process.start()
        process.join()
        code = process.exitcode
        if code is None:
            raise RuntimeError("Native worker did not terminate")
        output = (
            log.read_bytes().decode("utf-8", errors="replace") if log.exists() else ""
        )
        active_stage = stage_file.read_text() if stage_file.exists() else stage
        if code != 0 or not result.exists():
            raise error_type(
                "Native worker terminated unexpectedly",
                stage=active_stage,
                returncode=code,
                diagnostic=output[-8192:],
            )
        with result.open("rb") as source:
            kind, value = pickle.load(source)
        if kind == "error":
            assert isinstance(value, Exception)
            if isinstance(value, XtcError) and not value.diagnostic.strip():
                value.diagnostic = output[-8192:]
            elif output and not isinstance(value, XtcError):
                sys.stderr.write(output)
            raise value
        if kind == "unserializable":
            raise error_type(str(value), stage=active_stage, diagnostic=output[-8192:])
        if kind != "result":
            raise RuntimeError(f"Invalid native worker response: {kind}")
        if output:
            sys.stderr.write(output)
        return cast(T, value)
