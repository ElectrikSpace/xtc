#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any

PadConstraint = dict[str, Any]
# One list of constraints per buffer dimension (in schedule dimension order).
PadStrategies = list[list[PadConstraint]]
# As given by the user: a constraint or a list of constraints per dimension.
PadStrategiesArg = list[PadConstraint | list[PadConstraint]] | None

# Available padding constraints


def pad_divisible_by(value: int) -> PadConstraint:
    """
    Pad size is divisible by the given value
    """
    return {"type": "div_constraint", "args": [value]}

def pad_modulo_value(modulo: int, value: int) -> PadConstraint:
    """
    The pad size modulo a given value has a given value
    """
    assert value < modulo
    return {"type": "modulo_constraint", "args": [modulo, value]}


# Compute the padding size


def normalize_pad_strategies(
    strategies: PadStrategiesArg,
) -> PadStrategies | None:
    """Accept one constraint or a list of constraints per dimension."""
    if strategies is None:
        return None
    return [[s] if isinstance(s, dict) else list(s) for s in strategies]


def compute_padding(size: int, constraints: list[PadConstraint]) -> int:
    """Smallest size >= size satisfying the constraints applied in order."""
    pad = size
    for constraint in constraints:
        args = constraint["args"]
        if constraint["type"] == "div_constraint":
            pad += -pad % args[0]
        elif constraint["type"] == "modulo_constraint":
            pad += (args[1] - pad) % args[0]
        else:
            raise ValueError(f"unknown padding constraint: {constraint['type']}")
    return pad
