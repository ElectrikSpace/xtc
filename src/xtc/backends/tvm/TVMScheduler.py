#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import sys
from typing_extensions import override
from typing import TextIO, TypeAlias
from io import StringIO
import numpy as np

from xtc.itf.schd.scheduler import DEFAULT_ROOT
import xtc.backends.tvm as backend
import xtc.itf as itf

# Actual backend Schedule implementation is a mapping
# from op name to the TVM schedule string
ScheduleImpl: TypeAlias = dict[str, str]


class TVMScheduler(itf.schd.Scheduler):
    def __init__(
        self,
        backend: "backend.TVMBackend",
        nodes: list[str] | None = None,
        default_node: str | None = None,
    ) -> None:
        self._backend = backend
        if nodes is None:
            self._scheduled_ops = backend._ops
        else:
            self._scheduled_ops = {name: backend._ops[name] for name in nodes}
        assert len(self._scheduled_ops) > 0
        if default_node is None:
            candidate_ops = list(self._scheduled_ops.values())
        else:
            assert default_node in self._scheduled_ops
            candidate_ops = [
                v for k, v in self._scheduled_ops.items() if k == default_node
            ]
        self._op = candidate_ops[-1]
        self._abstract_dims = {d: d for d in self._op.operator.dims()}
        self._sizes = list(self._op.operator.dims_sizes().values())
        self.dims = list(self._op.operator.dims())
        self.tiles: dict[str, dict[str, int]] = {
            k: {k: v} for k, v in zip(self.dims, self._sizes)
        }
        self.permutation: list[str] = list(self.tiles.keys())
        self.vectorization: list[str] = []
        self.parallelization: list[str] = []
        self.unrolling: dict[str, int] = {}
        self.write_caches: list[str] = []
        self.read_buffers: list[tuple[str, int, bool]] = []
        self._update_loops()

    @property
    def dims_sizes(self) -> dict[str, int]:
        return {d: s for d, s in zip(self.dims, self._sizes)}

    @property
    def parallel_dims(self) -> list[str]:
        return [self._abstract_dims[d] for d in self._op.operator.dims("P")]

    @property
    def reduction_dims(self) -> list[str]:
        return [self._abstract_dims[d] for d in self._op.operator.dims("R")]

    @property
    @override
    def backend(self) -> itf.back.Backend:
        return self._backend

    @override
    def schedule(self) -> itf.schd.Schedule:
        io = StringIO()
        self._dump_tvm_schedule(outf=io)
        sched = io.getvalue()
        assert self._op.name is not None
        schedule_impl = {self._op.name: sched}
        return TVMSchedule(scheduler=self, schedule_impl=schedule_impl)

    def _update_loops(self):
        loops = dict()
        parallels = []
        for tile_level in range(len(max(self.tiles.values(), key=len))):
            for k, v in self.tiles.items():
                if tile_level >= len(v):
                    continue
                dim_name = list(v.keys())[tile_level]
                loops[dim_name] = v[dim_name]
                if k in self.parallel_dims:
                    parallels.append(dim_name)
        self._working_dims = loops
        self._working_parallel_dims = parallels
        self.permutation = list(self._working_dims.keys())

    @override
    def set_dims(self, dims: list[str]) -> None:
        assert len(dims) == len(self.dims)
        self.dims = dims[:]
        self._abstract_dims = {d: a for d, a in zip(self._op.operator.dims(), dims)}
        self.tiles = {k: {k: v} for k, v in zip(self.dims, self._sizes)}
        self.permutation = list(self.tiles.keys())

    @override
    def split(
        self, dim: str, segments: dict[str, int], root: str = DEFAULT_ROOT
    ) -> None: ...

    @override
    def tile(self, dim: str, tiles: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        all_tiles = {dim: self.dims_sizes[dim], **tiles}
        parent_tile_size = self.dims_sizes[dim]
        for name, size in all_tiles.items():
            assert size >= 1, f"unexpected tile size < 1 for axis {dim}"
            assert parent_tile_size >= size, (
                f"unexpected tile size < inner tile for axis {dim}"
            )
            self.tiles[dim][name] = size
            parent_tile_size = size
        self._update_loops()

    @override
    def interchange(self, permutation: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in self.permutation:
            assert axis in permutation, f"missing axis {axis} in interchange"
        for axis in permutation:
            assert axis in self.permutation, f"unexpected axis {axis} in interchange"
        self.permutation = permutation

    @override
    def buffer_at(
        self, axis: str, mtype: str | None = None, root: str = DEFAULT_ROOT
    ) -> None:
        assert mtype is None or mtype == "local"
        self.write_caches.append(axis)

    @override
    def pack_at(
        self,
        axis: str,
        input_idx: int,
        mtype: str | None = None,
        pad: bool = False,
        root: str = DEFAULT_ROOT,
    ) -> None:
        assert mtype is None or mtype == "local"
        assert input_idx >= 0 and input_idx < len(self._op.np_inputs_spec())
        self.read_buffers.append((axis, input_idx, pad))

    @override
    def vectorize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in axes:
            assert axis in self._working_parallel_dims, f"non parallel axis {axis}"
        self.vectorization = axes

    @override
    def parallelize(self, axes: list[str], root: str = DEFAULT_ROOT) -> None:
        for axis in axes:
            assert axis in self._working_parallel_dims, f"non parallel axis {axis}"
        self.parallelization = axes

    @override
    def unroll(self, unrolls: dict[str, int], root: str = DEFAULT_ROOT) -> None:
        for axis, unroll in unrolls.items():
            assert unroll > 0, f"unroll < 1 not supported for axis {axis}"
        self.unrolling = unrolls

    def _full_packs(self) -> dict[str, tuple[int, int, int]]:
        def factor_offset(input_idx: int, pad: bool):
            if not pad:
                return 0, 0
            input_spec = self._op.np_inputs_spec()[input_idx]
            if len(input_spec["shape"]) < 2:
                return 0, 0
            # Assume for CPU common number of sets and line size for L1
            # Except to minimize conflicts by setting the inner axis
            # size to a factor of num_sets and adding +1
            num_sets, line_size = 64, 64
            elt_size = np.dtype(input_spec["dtype"]).itemsize
            elts_per_line = line_size // elt_size
            return elts_per_line * num_sets, elts_per_line

        packs = {}
        for axis, input_idx, pad in self.read_buffers:
            factor, offset = factor_offset(input_idx, pad)
            packs[axis] = (input_idx, factor, offset)
        return packs

    def _full_order(self) -> list[str]:
        tiles_sizes = self._working_dims
        permutation = self.permutation
        unrolling = self.unrolling
        permutation_with_unrolls = []
        for axis in permutation:
            permutation_with_unrolls.append(axis)
            if axis in unrolling:
                if unrolling[axis] < tiles_sizes[axis]:
                    permutation_with_unrolls.append(f"__u_{axis}")
        return permutation_with_unrolls

    def _full_tiles(self) -> dict[str, dict[str, int]]:
        unrolling = self.unrolling
        tiles_with_unroll = {}
        for dim, dim_tiles in self.tiles.items():
            dim_tiles_with_unroll = {}
            for axis, size in dim_tiles.items():
                dim_tiles_with_unroll.update({axis: size})
                if axis in unrolling:
                    if unrolling[axis] < size:
                        dim_tiles_with_unroll.update({f"__u_{axis}": unrolling[axis]})
            tiles_with_unroll[dim] = dim_tiles_with_unroll
        return tiles_with_unroll

    def _full_tilings(self) -> dict[str, tuple[str, str, int]]:
        order = self._full_order()
        tiles = self._full_tiles()
        tilings = {}
        for dim, dim_tiles in tiles.items():
            t_axes = list(dim_tiles.keys())
            t_sizes = list(dim_tiles.values())
            tilings[dim] = (dim, "", t_sizes[0])
            for idx in range(1, len(dim_tiles)):
                tilings[t_axes[idx]] = (dim, t_axes[idx - 1], t_sizes[idx])
        tilings = {axis: tilings[axis] for axis in order}
        return tilings

    def _write_buffer_tiling(
        self, axis: str, tilings: dict[str, tuple[str, str, int]]
    ) -> tuple[dict[str, tuple[str, str, int]], dict[str, tuple[str, str, int]]]:
        child = {
            parent: (dim, axis, factor)
            for axis, (dim, parent, factor) in tilings.items()
        }
        tiles_axis = list(tilings)
        tile_idx = tiles_axis.index(axis)
        outer_tiles = dict(list(tilings.items())[: tile_idx + 1])
        inner_tiles = dict(list(tilings.items())[tile_idx + 1 :])
        outers_dims = set()
        for axis, (dim, parent, factor) in list(outer_tiles.items()):
            outers_dims.add(dim)
            factor = child.get(axis, ("", "", 1))[2]
            outer_tiles[f"{dim}_"] = (dim, axis, factor)
        dims = set()
        for axis, (dim, parent, factor) in list(inner_tiles.items()):
            if outers_dims and dim not in outers_dims:
                outers_dims.add(dim)
                if dim in self.parallel_dims:
                    outer_tiles[f"{dim}_"] = (dim, dim, 0)
            if dim not in dims:
                dims.add(dim)
                parent = ""
            inner_tiles[axis] = (dim, parent, factor)
        return outer_tiles, inner_tiles

    def _full_write_buffers(
        self,
    ) -> dict[tuple[str, str, str], dict[str, tuple[str, str, int]]]:
        tilings = self._full_tilings()
        reorder_idx = {axis: idx for idx, axis in enumerate(tilings)}
        write_axis = sorted(self.write_caches, key=lambda axis: reorder_idx[axis])
        buffer_tilings = {}
        out = ("O", "", "")
        tiling = tilings
        for idx, axis in enumerate(write_axis):
            outer_tiling, inner_tiling = self._write_buffer_tiling(axis, tiling)
            buffer_tilings[out] = outer_tiling
            out = (f"O_W{idx}", out[0], axis)
            tiling = inner_tiling
        buffer_tilings[out] = tiling
        return buffer_tilings

    def _emit_assign_axis(self, sch: str, tens: str, outf: TextIO) -> None:
        if self.parallel_dims:
            print(f"{', '.join(self.parallel_dims)}, = {tens}.op.axis", file=outf)
        if self.reduction_dims:
            print(
                f"{', '.join(self.reduction_dims)}, = {tens}.op.reduce_axis", file=outf
            )

    def _emit_assign_tilings(
        self,
        sch: str,
        tens: str,
        tilings: dict[str, tuple[str, str, int]],
        outf: TextIO,
    ) -> None:
        for axis, (dim, parent, factor) in tilings.items():
            if not parent:
                if axis != dim:
                    print(f"{axis} = {dim}", file=outf)
                continue
            if factor > 0:
                print(
                    f"{parent}, {axis} = {sch}[{tens}].split({parent}, factor={factor})",
                    file=outf,
                )
            else:
                print(f"{axis} = {parent}", file=outf)
        print(f"{sch}[{tens}].reorder({', '.join(tilings)})", file=outf)

    def _dump_tvm_schedule(
        self, obj: str = "obj", sch: str = "sch", outf: TextIO = sys.stdout
    ):
        tilings = self._full_write_buffers()
        packings = self._full_packs()
        if packings:
            print(f"INPS = list({obj}.values())[:-1]", file=outf)
        for (tens, parent, axis), tiles in tilings.items():
            if not parent:
                print(f"{tens} = {obj}['{self._op.name}']", file=outf)
            else:
                print(f'{tens} = {sch}.cache_write({parent}, "local")', file=outf)
            for tile_axis in tiles:
                if tile_axis in packings:
                    inp_idx, _, _ = packings[tile_axis]
                    print(
                        f'I_R{inp_idx} = {sch}.cache_read(INPS[{inp_idx}], "local", [{tens}])',
                        file=outf,
                    )
        for idx, ((tens, parent, axis), tiles) in enumerate(tilings.items()):
            if parent:
                print(f"{sch}[{tens}].compute_at({sch}[{parent}], {axis})", file=outf)
            self._emit_assign_axis(sch, tens, outf)
            self._emit_assign_tilings(sch, tens, tiles, outf)
            for tile_axis in tiles:
                if tile_axis in packings:
                    inp_idx, factor, offset = packings[tile_axis]
                    print(
                        f"{sch}[I_R{inp_idx}].compute_at({sch}[{tens}], {tile_axis})",
                        file=outf,
                    )
                    if factor != 0:
                        print(
                            f"{sch}[I_R{inp_idx}].storage_align(I_R{inp_idx}.op.axis[-2], factor={factor}, offset={offset})",
                            file=outf,
                        )
            for axis, unroll in self.unrolling.items():
                for u_axis in [f"__u_{axis}", axis]:
                    if u_axis in tiles:
                        print(f"{sch}[{tens}].unroll({u_axis})", file=outf)
                        break
            for axis in self.vectorization:
                if axis in tiles:
                    print(f"{sch}[{tens}].vectorize({axis})", file=outf)
            if self.parallelization:
                if self.parallelization[0] in tiles:
                    if len(self.parallelization) > 1:
                        print(
                            f"{self.parallelization[-1]} = {sch}[{tens}].fuse({', '.join(self.parallelization)})",
                            file=outf,
                        )
                    print(
                        f"{sch}[{tens}].parallel({self.parallelization[-1]})", file=outf
                    )

    def dump_schedule(self, obj: str | None = None, outf: TextIO = sys.stdout):
        if obj is None:
            obj = "sch"
        for dim, tiles in self.tiles.items():
            t_tiles = {k: v for i, (k, v) in enumerate(tiles.items()) if i >= 1}
            print(f"{obj}.tile('{dim}', {t_tiles})", file=outf)
        print(f"{obj}.interchange({self.permutation})", file=outf)
        print(f"{obj}.vectorize({self.vectorization})", file=outf)
        print(f"{obj}.unroll({self.unrolling})", file=outf)
        print(f"{obj}.parallelize({self.parallelization})", file=outf)

    def get_schedule_str(self, obj: str | None = None) -> str:
        io = StringIO()
        self.dump_schedule(outf=io)
        return io.getvalue()

    @override
    def __str__(self) -> str:
        return self.get_schedule_str()


class TVMSchedule(itf.schd.Schedule):
    def __init__(self, scheduler: "TVMScheduler", schedule_impl: ScheduleImpl) -> None:
        self._scheduler = scheduler
        self._schedule_impl = schedule_impl

    @property
    def schedule_impl(self) -> ScheduleImpl:
        return self._schedule_impl

    @property
    @override
    def scheduler(self) -> itf.schd.Scheduler:
        return self._scheduler

    @override
    def __str__(self) -> str:
        return "\n".join(self._schedule_impl.values())
