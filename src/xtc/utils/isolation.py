#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Process boundary for native code that can abort the interpreter."""

import logging
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

from xtc.errors import XtcError, XtcTimeoutError

T = TypeVar("T")
_stage_file: Path | None = None


def mark_isolated_stage(stage: str) -> None:
    """Record the active native stage for failures that cannot raise."""
    if _stage_file is not None:
        _stage_file.write_text(stage)


def _all_logging_handlers() -> list[logging.Handler]:
    loggers: list[logging.Logger] = [logging.getLogger()] + [
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    ]
    return [handler for logger in loggers for handler in logger.handlers]


def _worker(
    operation: Callable[[], T], log: Path, result: Path, stage_file: Path
) -> None:
    global _stage_file
    _stage_file = stage_file
    _, hard_limit = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard_limit))
    with log.open("wb", buffering=0) as output:
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        # Never flush the inherited sys.stdout/sys.stderr: when forked from a
        # thread pool worker, their locks may be held by another parent thread
        # (logging, progress bars) and the child would deadlock. Their pending
        # data belongs to the parent anyway.
        old_streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
        sys.stdout = open(1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)
        for handler in _all_logging_handlers():
            if (
                isinstance(handler, logging.StreamHandler)
                and handler.stream in old_streams
            ):
                handler.stream = sys.stderr
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
    # Skip the interpreter shutdown of the forked process: when forked from a
    # thread pool worker, joining the inherited (dead) pool threads fails.
    os._exit(0)


def run_isolated(
    operation: Callable[[], T],
    *,
    error_type: type[XtcError],
    stage: str,
    timeout: float | None = None,
) -> T:
    """Run native work in a forked process; return only serializable results.

    The caller must not depend on changes to in-memory objects in the worker.
    If `timeout` (seconds) expires, the worker is killed and XtcTimeoutError raised.
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
        process.join(timeout)
        if process.is_alive():
            process.kill()
            process.join()
            raise XtcTimeoutError(
                f"Native worker timed out after {timeout:g}s",
                stage=stage_file.read_text() if stage_file.exists() else stage,
            )
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
