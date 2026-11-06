#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from copy import deepcopy
from typing import Literal

from typing_extensions import override
from dataclasses import dataclass, asdict, field
from xtc.itf.schd.scheduler import DEFAULT_ROOT
from xtc.utils.pad import PadStrategiesArg
from xtc.schedules.loop_names import make_loop_name
from xtc.schedules.plain_schedule import PlainNodeSchedule, PlainNodeScheduler

__all__ = [
    "MlirNodeScheduler",
    "MlirNodeSchedule",
    "PackedBufferEntry",
]

ROOT_SEP = "/"

# Bare index: no padding request. ``(input_idx, True)``: padding slack is derived at
# compile time via :meth:`MlirTarget.pack_at_padding_heuristic`.
PackedBufferEntry = int | tuple[int, Literal[True]]


def basename(loop_name: str) -> str:
    return loop_name.split(ROOT_SEP)[-1]


@dataclass(frozen=True)
class MlirNodeSchedule(PlainNodeSchedule):
    dim_sizes: dict[str, int] | None = field(default=None, repr=False)
    read_rotates: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    reduction_dims: list[str] = field(default_factory=list, repr=False)


class MlirNodeScheduler:
    def __init__(
        self,
        node_name: str,
        node_ident: str,
        dims: list[str],
        loop_stamps: list[str] = [],
        dim_sizes: dict[str, int] | None = None,
        reduction_dims: list[str] | None = None,
    ) -> None:
        self._plain_sch = PlainNodeScheduler(
            node_name,
            node_ident,
            dims,
        )
        self._dim_sizes = dim_sizes
        self._reduction_dims = list(reduction_dims or [])
        self._read_rotates: dict[str, list[dict]] = {}

    @property
    def node_name(self) -> str:
        return self._plain_sch.node_name

    @property
    def node_ident(self) -> str:
        return self._plain_sch.node_ident

    def set_dims(self, dims: list[str]) -> None:
        self._plain_sch.set_dims(dims)

    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.split(dim, segments, root)

    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.tile(dim, tiles, root)

    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.interchange(permutation, root)

    def vectorize(
        self,
        axes: list[str] | dict[str, int | None],
        root: str = DEFAULT_ROOT,
    ) -> None:
        self._plain_sch.vectorize(axes, root)

    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.parallelize(axes, root)

    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.unroll(unrolls, root)

    def buffer_at(
        self,
        axis: str,
        mtype: str | None = None,
        root: str = DEFAULT_ROOT,
        pad: bool = False,
        pad_strategies: PadStrategiesArg = None,
    ) -> None:
        self._plain_sch.buffer_at(axis, mtype, root, pad, pad_strategies)

    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
        pad_strategies: PadStrategiesArg = None,
    ):
        self._plain_sch.pack_at(axis, input_idx, mtype, pad, root, pad_strategies)

    def define_memory_mesh(self, axes: dict[str, int]):
        self._plain_sch.define_memory_mesh(axes)

    def define_processor_mesh(self, axes: dict[str, int]):
        self._plain_sch.define_processor_mesh(axes)

    def distribute(self, axis: str, processor_axis: str, root: str = DEFAULT_ROOT):
        self._plain_sch.distribute(axis, processor_axis, root)

    def distributed_buffer_at(
        self,
        axis: str,
        input_idx: int,
        memory_axes: list[str],
        root: str = DEFAULT_ROOT,
    ):
        self._plain_sch.distributed_buffer_at(axis, input_idx, memory_axes, root)

    def read_rotate_at(
        self,
        loop: str,
        input_idx: int,
        axis: str,
        root: str = DEFAULT_ROOT,
    ):
        memory_mesh = self._plain_sch.memory_mesh
        assert memory_mesh, "Memory mesh has not been defined"
        assert axis in memory_mesh, "Memory axis not found in memory mesh"
        dist_axes = [
            dist["memory_axes"]
            for dists in self._plain_sch.distributed_buffers.values()
            for dist in dists
            if dist["input_idx"] == input_idx
        ]
        assert dist_axes, "read_rotate_at requires a distributed input buffer"
        assert any(axis in axes for axes in dist_axes), (
            "Input buffer is not distributed on the rotation axis"
        )
        axis_key = make_loop_name(root, loop)
        self._read_rotates.setdefault(axis_key, []).append(
            {"input_idx": input_idx, "axis": axis}
        )

    def fuse_producer_at(
        self, axis: str, input_idx: int, root: str = DEFAULT_ROOT
    ) -> None:
        self._plain_sch.fuse_producer_at(axis, input_idx, root)

    def fuse_consumer_at(self, axis: str, root: str = DEFAULT_ROOT) -> None:
        self._plain_sch.fuse_consumer_at(axis, root)

    def get_node_schedule(self) -> MlirNodeSchedule:
        plain_schedule = self._plain_sch.get_plain_schedule()
        return MlirNodeSchedule(
            **asdict(plain_schedule),
            dim_sizes=self._dim_sizes,
            read_rotates=deepcopy(self._read_rotates),
            reduction_dims=self._reduction_dims[:],
        )

    @override
    def __str__(self) -> str:
        return str(self.get_node_schedule())
