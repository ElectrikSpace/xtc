#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TypeAlias, Any
from typing_extensions import override
from collections.abc import Sequence, Mapping, Iterator, Generator
import itertools
import numpy as np

from xtc.itf.graph import Graph
from xtc.itf.schd import Scheduler
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.itf.search import Sample, Strategy
from xtc.schedules.descript import Descript
from xtc.utils.math import (
    factors_to_sizes,
    factors_enumeration,
)
from xtc.utils.algorithms import (
    sample_uniques,
)


__all__ = [
    "SDistStrategies",
    "SDistStrategyRegistration",
]

VecSample: TypeAlias = list[int]


@dataclass(frozen=True)
class SDistStrategyRegistration:
    cls: type[Strategy]
    default_args: list[Any] = field(default_factory=list)
    default_kwargs: dict[str, Any] = field(default_factory=dict)


class SDistBaseStrategy(Strategy):
    """Base abstract class for implementing the strategies in this file.

    All strategies in this file define the search space as set of samples
    which are 1-D int vectors of type VecSample.
    """

    def __init__(
        self,
        graph: Graph,
        sample_names: list[str] | None = None,
        vec_size: int = 16,
        max_unroll: int = 256,
        threads: int = 1,
        max_parallelize: int = -1,
        **kwargs: Any,
    ) -> None:
        self._graph = graph
        self._sample_names = sample_names
        self._vec_size = vec_size
        self._max_unroll = max_unroll
        self._threads = threads
        # Schedule output operation
        self._op = graph.outputs_nodes[0].operation
        self._stats: dict[str, int] = {}
        self._parallelize = self._threads > 1
        self._max_parallelize = max_parallelize
        self._vectorize = self._vec_size > 1
        self._unroll = self._max_unroll != 0
        # TODO: should go into some machine description
        self._arch_vreg_num = kwargs.get("vreg_num", 32)
        self._arch_l1_size = kwargs.get("l1_size", 32 * 1024)
        self._arch_l2_size = kwargs.get("l2_size", 1024 * 1024)

    @property
    @override
    def graph(self) -> Graph:
        return self._graph

    @property
    @override
    def sample_names(self) -> list[str]:
        assert self._sample_names is not None
        return self._sample_names

    @override
    def generate(self, scheduler: Scheduler, sample: Sample) -> None:
        # Ensure sample is valid list kind
        in_x = list(sample)
        self._generate(scheduler, in_x)

    @override
    def exhaustive(self) -> Iterator[VecSample]:
        return self._exhaustive()

    @override
    def default_schedule(self, opt_level: int = 2) -> VecSample:
        assert opt_level >= 0
        return self._default_schedule(opt_level)

    @override
    def sample(self, num: int, seed: int | None = 0) -> Iterator[VecSample]:
        assert num > 0
        assert seed is None or seed >= 0
        return self._sample(num, seed)

    @override
    def dict_to_sample(self, sample: dict[str, Any]) -> Sample:
        return list(sample.values())

    @override
    def sample_to_dict(self, sample: Sample) -> dict[str, int]:
        return dict(zip(self.sample_names, sample))

    @abstractmethod
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None: ...

    @abstractmethod
    def _default_schedule(self, opt_level: int) -> list[int]: ...

    def _exhaustive(self) -> Iterator[VecSample]:
        inds = self._independents()
        samples = self._iter_product(inds, stat="all")
        filtered = self._filter(samples)
        return filtered

    @property
    def stats(self) -> Mapping[str, int]:
        return self._stats

    def _constant_sizes(self) -> Mapping[str, int]:
        sizes = {a: v for a, v in self._op.dims.items() if isinstance(v, int)}
        return sizes

    def _iter_product(
        self, args: Sequence[Sequence[Sequence[int]]], stat: str | None = None
    ) -> Iterator[VecSample]:
        if stat:
            self._stats[stat] = 0
        for x in itertools.product(*args):
            if stat:
                self._stats[stat] += 1
            yield list(itertools.chain(*x))

    def _vector_axis(self) -> str | None:
        p_dims = list(self._op.dims_kind("P"))
        return p_dims[-1] if p_dims else None

    def _filter_unroll(
        self,
        indexes: list[int],
        v_index: int | None,
        samples: Iterator[VecSample],
        stat: str | None = None,
    ) -> Iterator[VecSample]:
        # Filter inner n_axes unrolled tiles if > max_unroll
        # assuming inner is vectorized
        if stat:
            self._stats[stat] = 0
        if self._max_unroll < 0:
            yield from samples
            return
        for x in samples:
            inners = np.array(x)[indexes]
            inner_unroll = np.prod(inners)
            vec_size = min(x[v_index] if v_index is not None else 1, self._vec_size)
            if inner_unroll / vec_size <= self._max_unroll:
                if stat:
                    self._stats[stat] += 1
                yield x

    def _sample_product(
        self, inds: list[list[list[int]]], num: int, rng: np.random.Generator
    ) -> list[VecSample]:
        draw = np.hstack(
            [
                np.array(var, dtype="int")[rng.integers(len(var), size=num)]
                for var in inds
            ]
        )
        return draw.tolist()

    def _sample(self, num: int, seed: int | None = 0) -> Iterator[VecSample]:
        rng = np.random.default_rng(seed=seed)
        inds = self._independents()

        def draw(num: int) -> Generator[tuple[int, ...]]:
            samples = self._sample_product(inds, num, rng)
            filtered = self._filter(iter(samples))
            return (tuple(sample) for sample in filtered)

        samples = sample_uniques(draw, num)
        return iter(list(sample) for sample in samples[:num])

    @abstractmethod
    def _independents(self) -> list[list[list[int]]]: ...

    @abstractmethod
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]: ...

class Strategy_SDist_Simple(SDistBaseStrategy):
    """Strategy for Simple SDist schedule without distribution.

    Each tiling parameter is free when exploring exhausive samples.
    """

    def __init__(self, graph: Graph, **kwargs: Any) -> None:
        super().__init__(
            graph,
            ["ic", "jc", "iv", "jv"],
            **kwargs,
        )

    @override
    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        ic, jc, iv, jv = in_x[:4]
        axes_order = ["i", "j", "i1", "j1", "iv", "jv", "k"]
        #axes_order = ["j", "i", "j1", "i1", "i2", "j2", "iv", "jv", "k"]
        vector_axes = ["iv", "jv", "k"]
        parallel_axes = []
        #if self._threads > 1:
        #    parallel_axes.append("j")
        sch.define_memory_mesh(axes={"mx": 1})
        sch.define_processor_mesh(axes={"px": 1, "psx": 1})
        print("TILE SIZES")
        print(ic)
        print(jc)
        sch.tile("i", {"i1": 256})
        sch.tile("j", {"j1": 256})
        #sch.tile("i", {"i1": ic})
        #sch.tile("j", {"j1": jc})
        assert ic > 8
        assert jc > 8
        sch.pack_at("i1", 1)
        #sch.pack_at("k", 1, pad=True)
        sch.pack_at("j1", 0)
        #sch.buffer_at("i1") FIXME bug with double buffering
        #sch.pack_at("i", 0, pad=True)
        #sch.tile("i", {"i1": iR * iL2, "i2": iR}, root=".")
        #sch.tile("j", {"j1": jR * jL3, "j2": jR}, root=".")
        #sch.tile("k", {"k1": kR1}, root=".")
        sch.tile("i", {"iv": 8}, root=".")
        sch.tile("j", {"jv": 8}, root=".")
        sch.interchange(axes_order, root=".")
        sch.distribute("j1", "psx")
        #sch.parallelize(parallel_axes, root=".")
        sch.vectorize(vector_axes, root=".")
        #sch.unroll(unroll_axes, root=".")

    @override
    def _independents(self) -> list[list[list[int]]]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = self._constant_sizes().values()
        tiles_i = factors_enumeration(i, 2)
        tiles_j = factors_enumeration(j, 2)
        tiles_k = factors_enumeration(k, 2)
        boolean = [[0], [1]]
        return [tiles_i, tiles_j, tiles_k, boolean, boolean]

    @override
    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
        v_index = 3
        indexes = [1, 3, 5]
        samples = self._filter_unroll(indexes, v_index, samples, stat="filtered")
        return samples

    @override
    def _default_schedule(self, opt_level: int) -> list[int]:
        # TODO: ref above, only support matmult like
        assert len(self._constant_sizes()) == 3
        i, j, k = i, j, k = self._constant_sizes().values()
        schedule = [1, 1, 1, 1, 1, 0, 0, 0]
        if opt_level >= 3:
            jtile = self._vec_size
            itile = 2  # TODO: IPC?
            ktile = 1
            idiv = i >= itile and i % itile == 0
            jdiv = j >= jtile and j % jtile == 0
            kdiv = k >= ktile and k % ktile == 0
            if idiv and jdiv and kdiv:
                schedule = [1, itile, 1, jtile, ktile, 2, 1, 1]
        return schedule


#class Strategy_GOTO(BaseStrategy):
#    """Strategy for Goto tiling with vectorisation.
#
#    Each tiling parameter is free when exploring exhausive samples.
#    """
#
#    def __init__(self, graph: Graph, **kwargs: Any) -> None:
#        super().__init__(
#            graph,
#            ["iL2", "iR", "jL3", "jR", "kL1", "unroll_k", "pack_B", "pack_A"],
#            **kwargs,
#        )
#
#    @override
#    def _generate(self, sch: Scheduler, in_x: list[int]) -> None:
#        # TODO: ref above, only support matmult like
#        assert len(self._constant_sizes()) == 3
#        iL2, iR, jL3, jR, kR1, unroll_k, pack_B, pack_A = in_x[:8]
#        axes_order = ["j", "k", "i", "j1", "i1", "k1", "i2", "j2"]
#        vector_axes = ["j2"]
#        parallel_axes = []
#        unroll_axes = {"i2": iR, "k1": unroll_k}
#        if self._threads > 1:
#            parallel_axes.append("j")
#        if pack_B:
#            sch.pack_at("k", 1, pad=True)
#        if pack_A:
#            sch.pack_at("i", 0, pad=True)
#        sch.tile("i", {"i1": iR * iL2, "i2": iR}, root=".")
#        sch.tile("j", {"j1": jR * jL3, "j2": jR}, root=".")
#        sch.tile("k", {"k1": kR1}, root=".")
#        sch.interchange(axes_order, root=".")
#        sch.parallelize(parallel_axes, root=".")
#        sch.vectorize(vector_axes, root=".")
#        sch.unroll(unroll_axes, root=".")
#
#    @override
#    def _independents(self) -> list[list[list[int]]]:
#        # TODO: ref above, only support matmult like
#        assert len(self._constant_sizes()) == 3
#        i, j, k = self._constant_sizes().values()
#        tiles_i = factors_enumeration(i, 2)
#        tiles_j = factors_enumeration(j, 2)
#        tiles_k = factors_enumeration(k, 2)
#        boolean = [[0], [1]]
#        return [tiles_i, tiles_j, tiles_k, boolean, boolean]
#
#    @override
#    def _filter(self, samples: Iterator[VecSample]) -> Iterator[VecSample]:
#        v_index = 3
#        indexes = [1, 3, 5]
#        samples = self._filter_unroll(indexes, v_index, samples, stat="filtered")
#        return samples
#
#    @override
#    def _default_schedule(self, opt_level: int) -> list[int]:
#        # TODO: ref above, only support matmult like
#        assert len(self._constant_sizes()) == 3
#        i, j, k = i, j, k = self._constant_sizes().values()
#        schedule = [1, 1, 1, 1, 1, 0, 0, 0]
#        if opt_level >= 3:
#            jtile = self._vec_size
#            itile = 2  # TODO: IPC?
#            ktile = 1
#            idiv = i >= itile and i % itile == 0
#            jdiv = j >= jtile and j % jtile == 0
#            kdiv = k >= ktile and k % ktile == 0
#            if idiv and jdiv and kdiv:
#                schedule = [1, itile, 1, jtile, ktile, 2, 1, 1]
#        return schedule

class SDistStrategies:
    _map: dict[str, SDistStrategyRegistration] = {}
    _aliases: dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_cls: type[Strategy],
        *,
        default_args: Sequence[Any] | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
        aliases: Sequence[str] = (),
    ) -> None:
        """Register a strategy name with its class and default arguments."""
        cls._map[name] = SDistStrategyRegistration(
            strategy_cls,
            list(default_args or []),
            dict(default_kwargs or {}),
        )
        for alias in aliases:
            cls.register_alias(alias, name)

    @classmethod
    def register_alias(cls, alias: str, name: str) -> None:
        cls._aliases[alias] = name

    @classmethod
    def names(cls, *, include_aliases: bool = False) -> Sequence[str]:
        names = list(cls._map.keys())
        if include_aliases:
            names += list(cls._aliases.keys())
        return names

    @classmethod
    def aliases(cls) -> Mapping[str, str]:
        return dict(cls._aliases)

    @classmethod
    def resolve_name(cls, name: str) -> str:
        seen: set[str] = set()
        while name in cls._aliases:
            if name in seen:
                raise ValueError(f"strategy alias cycle involving: {name}")
            seen.add(name)
            name = cls._aliases[name]
        return name

    @classmethod
    def registration(cls, name: str) -> SDistStrategyRegistration:
        name = cls.resolve_name(name)
        if name not in cls._map:
            raise ValueError(f"unknown strategy name: {name}")
        return cls._map[name]

    @classmethod
    def from_name(cls, name: str) -> type[Strategy]:
        return cls.registration(name).cls

    @classmethod
    def default_args(cls, name: str) -> list[Any]:
        return list(cls.registration(name).default_args)

    @classmethod
    def default_kwargs(cls, name: str) -> dict[str, Any]:
        return dict(cls.registration(name).default_kwargs)

    @classmethod
    def create(cls, name: str, graph: Graph, *args: Any, **kwargs: Any) -> Strategy:
        registration = cls.registration(name)
        all_args = {graph, *registration.default_args, *args}
        all_kwargs = {**registration.default_kwargs, **kwargs}
        return registration.cls(*all_args, **all_kwargs)


SDistStrategies.register("simple", Strategy_SDist_Simple)
