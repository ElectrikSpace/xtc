#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import operator
from functools import reduce
import numpy as np


_divisors_list_memo: dict[int, list[int]] = {}


def divisors_list(n: int) -> list[int]:
    """
    Returns the ordered list of divisors, including 1,
    for instance:
    divisors_list(1) = [1]
    divisors_list(6) = [1, 2, 3, 6]
    divisors_list(97) = [1, 97]
    divisors_list(112) = [1, 2, 4, 7, 8, 14, 16, 28, 56, 112]
    """
    if n in _divisors_list_memo:
        return _divisors_list_memo[n]
    factors = []
    step = 1 if n % 2 == 0 else 2
    for i in range(1, int(np.sqrt(n)) + 1, step):
        if n % i == 0:
            factors.append(i)
            if n // i != i:
                factors.append(n // i)
    factors.sort()
    _divisors_list_memo[n] = factors
    return factors


_factors_enumeration_memo: dict[tuple[int, int], list[list[int]]] = {}


def factors_enumeration(n: int, n_factors: int) -> list[list[int]]:
    """
    Returns all valid combinations of n_factors for a number
    such that the multiplied factors divides the number.
    factorization into n_factors.
    Note that the list is ordered with smaller factors firsts.
    For instance:
    factors_enumeration(1, 2): [[1,1]]
    factors_enumeration(97, 2): [[1,1], [1,97], [97,1]]
    factors_enumeration(28, 2): [[1, 1],[1, 2],[1, 4],[1, 7],[1, 14],[1, 28],
                                 [2, 1],[2, 2],[2, 7],[2, 14],
                                 [4, 1],[4, 7],[7, 1],[7, 2],[7, 4],
                                 [14, 1],[14, 2],[28, 1]]
    """
    if (n, n_factors) in _factors_enumeration_memo:
        return _factors_enumeration_memo[(n, n_factors)]
    enumeration = []
    current = [0] * n_factors

    def enumerate(level: int, remain: int) -> None:
        if level >= n_factors:
            enumeration.append(current.copy())
        else:
            for factor in divisors_list(remain):
                current[level] = factor
                enumerate(level + 1, remain // factor)

    enumerate(0, n)
    _factors_enumeration_memo[(n, n_factors)] = enumeration
    return enumeration


def sizes_to_factors(splits: list[int]) -> list[int]:
    """Convert an array of outer-inner sizes to outer-inner factors
    For instance:
    sizes_to_factors([]) -> []
    sizes_to_factors([8]) -> [8]
    sizes_to_factors([16, 4, 4, 2]) -> [4, 1, 2, 2]
    """
    red = lambda seq, x: seq + [x // reduce(operator.mul, seq, 1)]
    return list(reversed(reduce(red, reversed(splits), [])))


def factors_to_sizes(splits: list[int]) -> list[int]:
    """Convert an array of outer-inner factors to outer-inner sizes
    For instance:
    factors_to_sizes([]) -> []
    factors_to_sizes([8]) -> [8]
    factors_to_sizes([4, 1, 2, 2]) -> [16, 4, 4, 2]
    """
    red = lambda seq, x: seq + [x * seq[-1] if len(seq) > 0 else x]
    return list(reversed(reduce(red, reversed(splits), [])))


def mulall(args: list[int]) -> int:
    """Multiply all args in list"""
    return reduce(operator.mul, args, 1)


def pow2divisor(value: int) -> int:
    """Returns the largest power of 2 which divides value"""
    return value & -value


def estimate_unique_num_chao1(
    unique_num: int,
    once_num: int,
    twice_num: int,
) -> int:
    """
    Returns the Chao1 richness estimate in order to
    approximate the number of unique samples in a space from the
    already drawn unique num and the ones seen exactly once
    and twice. This gives a lower bound of the estimated uniques.
    Note that if unique_num is 0, the estimate is still 0.
    This should be used only for sufficiently large draws >= 20.
    We use the bias corrected version described in the SpadeR User guide.
    Ref: https://sites.google.com/view/chao-lab-website/software/spade
    Ref: https://osf.io/tb9w2/download
    """
    assert once_num >= 0
    assert twice_num >= 0
    assert unique_num >= once_num + twice_num
    chao1_bc = unique_num + (once_num * (once_num - 1)) / (2 * (twice_num + 1))
    return int(chao1_bc + 0.5)


def estimate_unique_prob_good_turing(
    sample_size: int,
    once_num: int,
) -> float:
    """
    Returns the probability estimate of a new unique from the current
    sample size (including possibly non uniques) and the ones seen exactly once.
    Smoothed in order to account for small or 0 samples size.
    Ref: https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation
    """
    assert sample_size >= 0
    assert sample_size >= once_num
    return estimate_count_prob_smooth(once_num, sample_size)


def estimate_count_prob_smooth(
    count: int,
    sample_size: int,
    alpha: float = 1.0,
) -> float:
    """
    Estimate a smoothed probability given the
    current count and sample size for an observation.
    Uses additive smoothing for handling low number or 0 values.
    Returns by convention 0 if alpha is 0 and samples is 0.
    Ref: https://en.wikipedia.org/wiki/Additive_smoothing
    """
    assert count <= sample_size
    assert alpha >= 0
    if alpha == 0 and sample_size == 0:
        return 0
    return (count + alpha) / (sample_size + alpha)
