#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""User-facing diagnostics for compilation and execution failures."""

import os
import shlex
import sys
from collections.abc import Sequence
from typing import TextIO

from typing_extensions import override


def colorize_error(message: str, *, stream: TextIO | None = None) -> str:
    if stream is None:
        stream = sys.stderr
    if "NO_COLOR" in os.environ or stream is None or not stream.isatty():
        return message
    return f"\033[38;5;208m{message}\033[0m"


class XtcError(RuntimeError):
    """A failure with actionable context and optional native diagnostics."""

    title = "XTC error"

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        command: Sequence[str] | None = None,
        returncode: int | None = None,
        diagnostic: str = "",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.command = tuple(command) if command is not None else None
        self.returncode = returncode
        self.diagnostic = diagnostic

    @override
    def __str__(self) -> str:
        lines = ["", self.title, f"  Stage: {self.stage}", f"  Error: {self.args[0]}"]
        if self.command is not None:
            lines.append(f"  Command: {shlex.join(self.command)}")
        if self.returncode is not None:
            if self.returncode < 0:
                lines.append(f"  Signal: {-self.returncode}")
            else:
                lines.append(f"  Exit status: {self.returncode}")
        if self.diagnostic.strip():
            lines.append("  Diagnostic:")
            lines.extend(f"    {line}" for line in self.diagnostic.strip().splitlines())
        return colorize_error("\n".join(lines))

    @override
    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            type(self)._restore,
            (self.args[0], self.stage, self.command, self.returncode, self.diagnostic),
        )

    @classmethod
    def _restore(
        cls,
        message: str,
        stage: str,
        command: tuple[str, ...] | None,
        returncode: int | None,
        diagnostic: str,
    ) -> "XtcError":
        return cls(
            message,
            stage=stage,
            command=command,
            returncode=returncode,
            diagnostic=diagnostic,
        )


class XtcCompileError(XtcError):
    """Compilation failed."""

    title = "XTC compilation failed"


class XtcRuntimeError(XtcError):
    """Native execution failed."""

    title = "XTC execution failed"
