#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
"""Schedule-related exceptions."""

import os
import sys

from typing_extensions import override


class ScheduleParseError(RuntimeError):
    """Raised when schedule parsing fails."""

    pass


class ScheduleInterpretError(RuntimeError):
    """Raised when schedule interpretation fails."""

    pass


class ScheduleValidationError(RuntimeError):
    """Raised when schedule validation fails."""

    def __init__(
        self,
        message: str,
        *,
        node: str | None = None,
        root: str | None = None,
        dimension: str | None = None,
    ) -> None:
        super().__init__(message)
        self.node = node
        self.root = root
        self.dimension = dimension

    @override
    def __str__(self) -> str:
        if self.node is None:
            return super().__str__()

        lines = ["", "Invalid XTC schedule", f"  Node: {self.node}"]
        if self.root is not None:
            lines.append(f"  Root: {self.root}")
        if self.dimension is not None:
            lines.append(f"  Dimension: {self.dimension}")
        lines.append(f"  Error: {super().__str__()}")
        message = "\n".join(lines)
        if "NO_COLOR" in os.environ or not sys.stderr.isatty():
            return message
        return f"\033[38;5;208m{message}\033[0m"
