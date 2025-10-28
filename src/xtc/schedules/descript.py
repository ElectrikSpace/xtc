#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any, Tuple
from dataclasses import dataclass
import re
from xtc.itf.schd.scheduler import Scheduler

SchedDict = dict[str, Any]


def descript_scheduler(
    scheduler: Scheduler,
    node_name: str,
    abstract_axis: list[str],
    spec: dict[str, dict],
):
    descript = Descript(scheduler=scheduler, abstract_axis=abstract_axis)
    descript.apply(node_name=node_name, spec=spec)


@dataclass(frozen=True)
class Descript:
    scheduler: Scheduler
    abstract_axis: list[str]

    def apply(self, node_name: str, spec: dict[str, dict]):
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec, head=[])
        self._check_flattened_schedule(flat_schedules)
        self._check_sizes(flat_schedules)

        self.scheduler.set_dims(self.abstract_axis)
        for schedule in flat_schedules:
            self._check_unroll_parameter_domain(schedule)
            self._check_tile_parameter_domain(schedule)
            self._check_unrolling_tiling(schedule)

            root = schedule["root"]

            for d, s in schedule["splits"].items():
                self.scheduler.split(d, s, root=root)

            for d, s in schedule["tiles"].items():
                self.scheduler.tile(d, s, root=root)

            self.scheduler.interchange(schedule["interchange"], root=root)
            self.scheduler.vectorize(schedule["vectorize"], root=root)
            self.scheduler.parallelize(schedule["parallelize"], root=root)
            self.scheduler.unroll(schedule["unroll"], root=root)

    def _flatten_schedule(
        self, root: str, spec: dict[str, dict], head: list[str]
    ) -> list[SchedDict]:
        recursive_scheds: list[SchedDict] = []
        sched: SchedDict = {
            "root": root,
            "splits": {},
            "tiles": {a: {} for a in self.abstract_axis},
            "interchange": [],
            "vectorize": [],
            "parallelize": [],
            "unroll": {},
        }
        # State of the schedule
        sizes: dict[str, int | None] = {}
        previous_cut: dict[str, int | None] = {a: 0 for a in self.abstract_axis}
        interchange: list[str] = head
        # Processing the schedule
        for declaration, val in spec.items():
            # Splits
            if ":" in declaration:
                axis_name, x, y = parse_split_declaration(declaration)
                self._check_axis_existence(axis_name)

                # The only declaration where y (the cut) is None is the
                # last one, so it cannot be the previous one.
                cut = previous_cut[axis_name]

                # When x (the starting point of the slice), is not
                # specified, it is the previous cut
                if x is None:
                    x = cut

                self._check_splitting_intervals(declaration, axis_name, cut, x, y)

                # Update the previous cut
                previous_cut[axis_name] = y
                # Save the cutting points of the new dimensions
                if not axis_name in sched["splits"]:
                    sched["splits"][axis_name] = {}
                new_dim_index = len(sched["splits"][axis_name])
                new_dim_name = f"{axis_name}[{new_dim_index}]"
                new_root_name = f"{root}/{new_dim_name}"
                sched["splits"][axis_name][new_dim_name] = x
                interchange.append(new_dim_name)
                # Fetch the schedule associated with the new dimension
                next_schedule = val
                assert isinstance(next_schedule, dict)
                inner_scheds = self._flatten_schedule(
                    spec=next_schedule, root=new_root_name, head=[axis_name]
                )
                recursive_scheds += inner_scheds
                continue

            # Tiles
            elif "#" in declaration:
                axis_name, tile_size = declaration.split("#")
                self._check_axis_existence(axis_name)
                try:
                    loop_size = int(tile_size)
                except:
                    raise Exception(
                        f"Invalid tile size: '{tile_size}' in {declaration}"
                    )

                tile_num = len(sched["tiles"][axis_name])
                loop_name = f"{axis_name}{tile_num}"
                sched["tiles"][axis_name][loop_name] = loop_size
                sizes[loop_name] = loop_size
                interchange.append(loop_name)

            elif declaration in self.abstract_axis:
                if declaration in interchange:
                    raise Exception(
                        f"""
                        Axis {declaration} is scheduled twice (or more).
                        """
                    )
                loop_name = declaration
                interchange.append(loop_name)

            else:
                self._unknown_axis_error(declaration)

            annotate(loop_name=loop_name, sizes=sizes, annotations=val, sched=sched)

        # Check if the last cut of each axis is either 0 or None.
        # None correspond to "until the end of the loop". 0 is the
        # default value, if it has 0 then it means the axis isn't splitted.
        # Any other value means the split is let in a partial state.
        for axis, cut in previous_cut.items():
            if cut is not None and cut != 0:
                raise Exception(
                    f"Splitting on axis {axis} should end but stops at {cut}"
                )

        sched["interchange"] = interchange
        return [sched] + recursive_scheds

    def _check_flattened_schedule(self, flat_schedules: list[dict[str, Any]]) -> None:
        tiles_to_axis: dict[str, str] = {}
        splits_to_axis: dict[str, str] = {}
        dims: list[str] = []
        seen_axes: dict[str, int | None] = {}
        for sched in flat_schedules:
            tiles = sched["tiles"]
            interchange = sched["interchange"]
            unrolls = sched["unroll"]
            vectorize = sched["vectorize"]

            tiles_to_axis.update(self._get_tiles_to_axis(sched))
            splits_to_axis.update(self._get_splits_to_axis(sched))
            dims += self._get_axis(sched)
            loops_to_axis = tiles_to_axis | splits_to_axis | dict(zip(dims, dims))
            vect_above = False

            for loop_name in interchange:
                # Vectorization inconsistencies
                if loop_name in vectorize:
                    vect_above = True
                elif vect_above:
                    raise Exception(
                        f"Inner loop {loop_name} isn't vectorized but an outer one is."
                    )
                # Tiling inconsistencies (odd sizes, use before define)
                if loop_name in dims:
                    seen_axes[loop_name] = None
                elif loop_name in tiles_to_axis:
                    axis = tiles_to_axis[loop_name]
                    if axis not in seen_axes:
                        raise Exception(
                            f"""
                            Axis '{axis}' must be defined before tiling can produce loop '{loop_name}'.
                            """
                        )
                    old_tile_size = seen_axes[axis]
                    tile_size = tiles[axis][loop_name]
                    if old_tile_size is not None and tile_size > old_tile_size:
                        raise Exception(
                            f"""
                            Inner tile {loop_name} on axis {axis} must be smaller than outer loop.
                            """
                        )
                    seen_axes[axis] = tile_size

            # Full unrolling needs tile size
            for loop_name in unrolls:
                axis = loops_to_axis[loop_name]
                if len(tiles[axis]) == 0 and unrolls[loop_name] == None:
                    raise Exception(
                        f"""
                        {axis} cannot be implicitly fully unrolled on an axis
                        that isn't tiled (needs an unroll factor)
                        """
                    )

        for dim in self.abstract_axis:
            if dim not in dims:
                raise Exception(f"{dim} defined but never used")

    def _check_sizes(self, flat_sched: list[dict[str, Any]]):
        self._check_sizes_aux(flat_sched, {})

    def _check_sizes_aux(
        self,
        flat_sched: list[dict[str, Any]],
        current_size_of_dim: dict[str, int | None] = {},
    ):
        if len(flat_sched) == 0:
            return
        sched = flat_sched[0]
        splits = sched["splits"]
        tiles = sched["tiles"]
        interchange = sched["interchange"]

        tiles_to_axis = self._get_tiles_to_axis(sched)
        splits_to_axis = self._get_splits_to_axis(sched)
        dims = self._get_axis(sched)
        loops_to_axis = tiles_to_axis | splits_to_axis | dict(zip(dims, dims))

        splits_to_sizes: dict[str, int] = {}
        for axis in splits:
            last_start = None
            for loop_name, start in reversed(splits[axis].items()):
                if last_start is not None:
                    size_of_split = last_start - start
                    splits_to_sizes[loop_name] = size_of_split
                last_start = start

        for loop_name in interchange:
            axis = loops_to_axis[loop_name]
            if loop_name in dims:
                if loop_name not in current_size_of_dim:
                    current_size_of_dim[loop_name] = None
            elif loop_name in tiles_to_axis:
                loop_size = tiles[axis][loop_name]
                old_dim_size = current_size_of_dim[axis]
                if old_dim_size is not None and loop_size > old_dim_size:
                    raise Exception(
                        f"""
                        Inner loop {loop_name} on axis {axis} must be smaller than outer loop.
                        """
                    )
            elif loop_name in splits_to_axis:
                new_current_size_of_dim = current_size_of_dim.copy()
                if loop_name in splits_to_sizes:
                    loop_size = splits_to_sizes[loop_name]
                    new_current_size_of_dim[axis] = loop_size
                else:
                    new_current_size_of_dim[axis] = None
                self._check_sizes_aux(
                    flat_sched=flat_sched.copy()[1:],
                    current_size_of_dim=new_current_size_of_dim,
                )

    def _get_axis(self, sched: dict[str, Any]) -> list[str]:
        refined_loops = list(self._get_tiles_to_axis(sched)) + list(
            self._get_splits_to_axis(sched)
        )
        return [loop for loop in sched["interchange"] if loop not in refined_loops]

    def _get_tiles_to_axis(self, sched: dict[str, Any]) -> dict[str, str]:
        return self._get_subloops_to_axis("tiles", sched)

    def _get_splits_to_axis(self, sched: dict[str, Any]) -> dict[str, str]:
        return self._get_subloops_to_axis("splits", sched)

    def _get_subloops_to_axis(self, key: str, sched: dict[str, Any]) -> dict[str, str]:
        tiles = sched[key]
        loop_to_axis: dict[str, str] = {}
        for axis_name, subloops in tiles.items():
            for loop_name in subloops:
                loop_to_axis[loop_name] = axis_name
        return loop_to_axis

    def _check_splitting_intervals(
        self,
        declaration: str,
        axis_name: str,
        cut: int | None,
        x: int | None,
        y: int | None,
    ):
        if cut is None:
            raise Exception(
                f"""
                {declaration} is defined on an already covered axis.
                This might be caused by a missing endpoint: {axis_name}
                """
            )

        assert isinstance(cut, int)
        assert isinstance(x, int)

        if x > cut:
            raise Exception(
                f"""
                Splitting doesn't cover the whole axis
                (jumps from {cut} to {x} on axis {axis_name})
                """
            )
        elif x < cut:
            raise Exception(
                f"""
                Splitting are overlapping on axis {axis_name}
                (covered until {cut} but restart at {x})
                """
            )

        assert x is not None

        if y is not None and x >= y:
            raise Exception(
                f"""
                Starting point in the splitting cannot be greater or equal to
                the ending point in: {declaration}
                """
            )

    def _check_unroll_parameter_domain(self, sched: dict[str, Any]):
        """Procedure that check if the unroll parameters domains are correct
        An unroll parameter should be strictly positive"""
        unroll = sched["unroll"]
        for axis, param in unroll.items():
            if param is not None and param <= 0:
                raise Exception(
                    f"""
                    Unroll parameter should be strictly positive:
                    \"{axis}\" = {{\"unroll\" = {param}}}.
                    """
                )

    def _check_tile_parameter_domain(self, sched: dict[str, Any]):
        """Procedure that check if the tiles parameters domains are correct
        An tile parameter should be strictly positive"""
        tiles = sched["tiles"]
        for axis, tile in tiles.items():
            for param in tile.values():
                if param <= 0:
                    raise Exception(
                        f"""
                        Tile sizes should be strictly positive:
                        \"{axis}#{param}\".
                        """
                    )

    def _unknown_axis_error(self, axis: str):
        raise Exception(
            f"""
            Axis {axis} is not a defined axis (defined axis: {self.abstract_axis}).
            """
        )

    def _check_axis_existence(self, axis: str):
        if axis not in self.abstract_axis:
            self._unknown_axis_error(axis)

    def _check_unrolling_tiling(self, sched: dict[str, Any]) -> None:
        """Procedure that check if an unrolled axis fits in the tile"""
        tiles = sched["tiles"]
        unrolls = sched["unroll"]

        for subaxis in tiles.values():
            for subaxis_name, tile_size in subaxis.items():
                # if the axis is unrolled and tiled and the unroll factor is
                # greater then the tile size
                if (
                    subaxis_name in unrolls
                    and tile_size > 1
                    and unrolls[subaxis_name] > tile_size
                ):
                    times = unrolls[subaxis_name]
                    raise Exception(
                        f"""
                        {subaxis_name} cannot be unrolled {times} times
                        on a tile of size {tile_size}
                        """
                    )

    def _check_axis_usage(
        self, sched: dict[str, Any], loop_to_axis: dict[str, str], unused_axis: set[str]
    ):
        interchange: list[str] = sched["interchange"]
        for loop_name in interchange:
            axis = loop_to_axis[loop_name]
            if axis in unused_axis:  # Remove used axis from unused set
                unused_axis.remove(axis)


def annotate(
    loop_name: str,
    sizes: dict[str, int | None],
    annotations: dict[str, Any],
    sched: dict[str, Any],
):
    for instr, param in annotations.items():
        assert isinstance(instr, str)
        assert isinstance(param, int | None)
        match instr:
            case "unroll":
                ufactor = (
                    sizes[loop_name] if param is None and loop_name in sizes else param
                )
                sched["unroll"][loop_name] = ufactor
            case "vectorize":
                if param is not None:
                    raise Exception(
                        f"Vectorize should not have a parameter (Feature not implemented)"
                    )
                sched["vectorize"].append(loop_name)

            case "parallelize":
                if param is not None:
                    raise Exception(
                        f"Parallelize should not have a parameter (Feature not implemented)"
                    )

                sched["parallelize"].append(loop_name)

            case _:
                raise Exception(f"Unknown annotation on {loop_name}: {instr}")


def parse_split_declaration(declaration: str) -> Tuple[str, int | None, int | None]:
    pattern = r"^(.*)\[(?:(-\d+|\d*)?):(?:(-\d+|\d*)?)\]$"
    match = re.match(pattern, declaration)
    if not match:
        raise Exception(f"Wrong format {declaration}")

    prefix, x_str, y_str = match.groups()
    x = int(x_str) if x_str else None
    y = int(y_str) if y_str else None
    return prefix, x, y
