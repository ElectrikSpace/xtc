#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from math import lcm
from typing import Any

PadConstraint = dict[str, Any]
# One list of constraints per buffer dimension (in schedule dimension order).
PadStrategies = list[list[PadConstraint]]
# As given by the user: a constraint or a list of constraints per dimension.
PadStrategiesArg = list[PadConstraint | list[PadConstraint]] | None

# Available padding constraints.
#
# A constraint with ``zero=True`` is a *compute* padding: the padded elements
# are filled with zeros and the computation is extended on them (e.g. padding
# to a multiple of the vector/tile size). A constraint with ``zero=False`` is a
# *layout* padding: only the allocation (hence the strides) is enlarged, the
# padded elements are neither initialized nor computed on (e.g. avoiding cache
# set or bank conflicts).


def pad_divisible_by(value: int, zero: bool = False) -> PadConstraint:
    """
    Pad size is divisible by the given value
    """
    assert value > 0
    return {"type": "div_constraint", "args": [value], "zero": zero}


def pad_modulo_value(modulo: int, value: int, zero: bool = False) -> PadConstraint:
    """
    The pad size modulo a given value has a given value
    """
    assert 0 <= value < modulo
    return {"type": "modulo_constraint", "args": [modulo, value], "zero": zero}


# Strategy for dimensions whose partial tiles are padded to the full tile
# (``sdist.pad_partial_tile``): the loops are padded and the accesses past the
# bounds are guarded (clamped DMAs), the padding is not zero filled by the
# schedule. Without it, partial tiles are dynamic and not supported by every
# backend.
pad_partial_tiles: PadConstraint = {
    "type": "partial_tiles",
    "args": [],
    "zero": False,
}


def has_pad_partial_tiles(strategies: PadStrategies | None) -> bool:
    return strategies is not None and any(
        c["type"] == "partial_tiles" for cs in strategies for c in cs
    )


def needs_zero_fill(strategies: PadStrategies | None) -> bool:
    """Without strategies, the default padding is zero filled."""
    return strategies is None or any(
        c.get("zero", False) for cs in strategies for c in cs
    )


def zero_multiple(constraints: list[PadConstraint]) -> int:
    """Multiple the extent is zero filled to, by the ``zero=True`` divisibility
    constraints (1 if none)."""
    return lcm(
        *(
            int(c["args"][0])
            for c in constraints
            if c["type"] == "div_constraint" and c.get("zero", False)
        )
    )


# Compute the padding size


def normalize_pad_strategies(
    strategies: PadStrategiesArg,
) -> PadStrategies | None:
    """Accept one constraint or a list of constraints per dimension."""
    if strategies is None:
        return None
    return [[s] if isinstance(s, dict) else list(s) for s in strategies]


def _modulus(constraint: PadConstraint) -> int:
    if constraint["type"] in ("div_constraint", "modulo_constraint"):
        return int(constraint["args"][0])
    if constraint["type"] == "partial_tiles":
        return 1
    raise ValueError(f"unknown padding constraint: {constraint['type']}")


def _satisfies(size: int, constraint: PadConstraint) -> bool:
    args = constraint["args"]
    if constraint["type"] == "partial_tiles":
        return True
    if constraint["type"] == "div_constraint":
        return size % args[0] == 0
    if constraint["type"] == "modulo_constraint":
        return size % args[0] == args[1]
    raise ValueError(f"unknown padding constraint: {constraint['type']}")


def _smallest_satisfying(size: int, constraints: list[PadConstraint]) -> int:
    """Smallest value >= size satisfying all the constraints."""
    period = lcm(*(_modulus(c) for c in constraints)) if constraints else 1
    for candidate in range(size, size + period):
        if all(_satisfies(candidate, c) for c in constraints):
            return candidate
    raise ValueError(f"incompatible padding constraints: {constraints}")


def compute_padding(size: int, constraints: list[PadConstraint]) -> tuple[int, int]:
    """
    Returns ``(fill, alloc)`` for a dimension of extent ``size``:
    - ``fill``: smallest extent >= size satisfying the ``zero=True``
      constraints, the elements in ``[size, fill)`` are zero filled and
      computed on;
    - ``alloc``: smallest allocated extent >= fill satisfying all the
      constraints, the elements in ``[fill, alloc)`` only change the strides.
    """
    zero = [c for c in constraints if c.get("zero", False)]
    fill = _smallest_satisfying(size, zero)
    alloc = _smallest_satisfying(fill, constraints)
    return fill, alloc
