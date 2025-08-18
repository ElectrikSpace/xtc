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
        flat_schedules = self._flatten_schedule(root=node_name, spec=spec)
        self._check_flattened_schedule(flat_schedules)
        for schedule in flat_schedules:
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
        self,
        root: str,
        spec: dict[str, dict],
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
        interchange: list[str] = []
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
                    spec=next_schedule, root=new_root_name
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
                loop_name = declaration
                interchange.append(loop_name)

            else:
                raise Exception(
                    f"Axis {declaration} is not a defined axis. Known axis are: {self.abstract_axis}"
                )

            annotate(loop_name=loop_name, sizes=sizes, annotations=val, sched=sched)

        # Check if the last cut of each axis is either 0 or None
        for axis, cut in previous_cut.items():
            if (
                cut is not None and cut != 0
            ):  # None correspond to "until the end of the loop". 0 is the default value, if it has 0 then it means the axis isn't splitted
                raise Exception(
                    f"Splitting on axis {axis} should end but stops at {cut}"
                )

        sched["interchange"] = interchange
        return [sched] + recursive_scheds

    def _check_flattened_schedule(self, flat_schedules: list[dict[str, Any]]):
        """Procedure that perform multiple checks on a flattened schedule
        If one of them fails, it raises an error"""

        loop_to_axis: dict[str, str] = dict()
        for sched in flat_schedules:
            node_name = sched["root"]

            for key in sched["tiles"]:
                for value in sched["tiles"][key]:
                    loop_to_axis[value] = key
                loop_to_axis[key] = key

            for key in sched["splits"]:
                for value in sched["splits"][key]:
                    loop_to_axis[value] = key
                loop_to_axis[key] = key

            self._check_unroll_parameter_domain(sched)
            self._check_tile_parameter_domain(sched)

            self._check_unrolling_tiling(sched)
            self._check_no_tile_full_unroll(sched, loop_to_axis)
            self._check_axis_definition(sched, loop_to_axis)

        self._check_splits(
            flat_schedules,
            loop_to_axis,
            {e for e in self.abstract_axis},
            set(),
            dict(),
            dict(),
            0,
            0,
        )

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
                f"{declaration} is defined on an already covered axis. This might be caused by a missing endpoint: {axis_name}"
            )

        assert isinstance(cut, int)
        assert isinstance(x, int)

        if x > cut:
            raise Exception(
                f"Splitting doesn't cover the whole axis (jumps from {cut} to {x} on axis {axis_name})"
            )
        elif x < cut:
            raise Exception(
                f"Splitting are overlapping on axis {axis_name} (covered until {cut} but restart at {x})"
            )

        assert x is not None

        if y is not None and x >= y:
            raise Exception(
                f"Starting point in the splitting cannot be greater or equal to the ending point in: {declaration}"
            )

    def _check_unroll_parameter_domain(self, sched: dict[str, Any]):
        """Procedure that check if the unroll parameters domains are correct
        An unroll parameter should be strictly positive"""
        unroll = sched["unroll"]
        for axis, param in unroll.items():
            if param is not None and param <= 0:
                raise Exception(
                    f'Unroll parameter should be strictly positive: "{axis}" = {{"unroll" = {param}}}.'
                )

    def _check_tile_parameter_domain(self, sched: dict[str, Any]):
        """Procedure that check if the tiles parameters domains are correct
        An tile parameter should be strictly positive"""
        tiles = sched["tiles"]
        for axis, tile in tiles.items():
            for param in tile.values():
                if param <= 0:
                    raise Exception(
                        f'Tile sizes should be strictly positive: "{axis}#{param}".'
                    )

    def _check_axis_existence(self, axis: str):
        if axis not in self.abstract_axis:  # axis not defined
            raise Exception(
                f"Axis {axis} is not a defined axis. Defined axis are: {self.abstract_axis}"
            )

    def _check_axis_definition(
        self, sched: dict[str, Any], loop_to_axis: dict[str, str]
    ):
        """Procedure that check if the axis used are defined and remove the used one from the unused set"""
        interchange = sched["interchange"]
        for loop_name in interchange:
            axis = loop_to_axis[loop_name]
            self._check_axis_existence(axis)

    def _check_unrolling_tiling(self, sched: dict[str, Any]) -> None:
        """Procedure that check if an unrolled axis fits in the tile"""
        tiles = sched["tiles"]
        unrolls = sched["unroll"]

        for _, subaxis in tiles.items():
            for subaxis_name, tile_size in subaxis.items():
                # if the axis is unrolled and tiled and the unroll factor is greater then the tile size
                if (
                    subaxis_name in unrolls
                    and tile_size > 1
                    and unrolls[subaxis_name] > tile_size
                ):
                    raise Exception(
                        f"{subaxis_name} cannot be unrolled {unrolls[subaxis_name]} times on a tile of size {tile_size}"
                    )

    def _check_no_tile_full_unroll(
        self, sched: dict[str, Any], loop_to_axis: dict[str, str]
    ) -> None:
        tiles = sched["tiles"]
        unrolls = sched["unroll"]

        for loop_name in unrolls:
            axis = loop_to_axis[loop_name]
            if tiles[axis] == dict() and unrolls[loop_name] == None:
                raise Exception(
                    f"{axis} cannot be implicitly fully unrolled on an axis that isn't tiled (needs an unroll factor)"
                )

    def _check_splits(
        self,
        flat_sched: list[dict[str, Any]],
        loop_to_axis: dict[str, str],
        unused_axis: set[str],
        knowned_vectorized_axis: set[str] = set(),
        last_sizes: dict[str, int] = dict(),
        current_split_size: dict[str, int] = dict(),
        sched_index: int = 0,
        depth: int = 0,
    ) -> int:
        """Procedure that check the vectorization of each axis"""
        sched = flat_sched[sched_index]
        splits = sched["splits"]

        self._check_vectorize_inner_tile(sched, loop_to_axis, knowned_vectorized_axis)
        self._check_tile_divisible(sched, last_sizes, current_split_size)
        self._check_axis_usage(sched, loop_to_axis, unused_axis)

        sub_split_sizes = []
        for axis in splits:
            sub_split_size = dict()
            last_start = None
            for split, start in splits[axis].items():
                if last_start is not None:
                    sub_split_size[axis] = start - last_start
                last_start = start

            sub_split_sizes.append(sub_split_size)
            sub_split_sizes.append(dict())

        nbcall = 0
        unused_axis_copy = {e for e in unused_axis}
        for axis in splits.values():
            for i in range(len(axis)):
                nbcall += 1
                unused_axis_copy = {e for e in unused_axis}
                sched_index += self._check_splits(
                    flat_sched,
                    loop_to_axis,
                    unused_axis_copy,
                    {e for e in knowned_vectorized_axis},
                    {key: value for key, value in last_sizes.items()},
                    sub_split_sizes[i],
                    i + sched_index + 1,
                    depth + 1,
                )

        if unused_axis_copy != set():  # Some axis are not used
            raise Exception(f"{unused_axis_copy} defined but never used")

        return nbcall

    def _check_vectorize_inner_tile(
        self,
        sched: dict[str, Any],
        loop_to_axis: dict[str, str],
        knowned_vectorized_axis: set[str] = set(),
    ) -> None:
        """Procedure that check the vectorization of each axis"""
        vectorize: list[str] = sched["vectorize"]
        interchange: list[str] = sched["interchange"]

        for loop_name in interchange:
            axis = loop_to_axis[loop_name]
            if loop_name in vectorize and axis not in knowned_vectorized_axis:
                knowned_vectorized_axis.add(axis)

            elif loop_name not in vectorize and axis in knowned_vectorized_axis:
                raise Exception(
                    f"Inner loop on {axis} isn't vectorized but an outer one is."
                )

    def _check_tile_divisible(
        self,
        sched: dict[str, Any],
        last_sizes: dict[str, int],
        current_split_size: dict[str, int] = dict(),
    ) -> None:
        """Procedure that check for each axis if the inner tiles are dividers of outer tiles"""
        tiles = sched["tiles"]

        for axis, tile in tiles.items():
            for tile_size in tile.values():
                if axis not in last_sizes.keys():  # First tile tile_size found
                    last_sizes[axis] = tile_size

                elif last_sizes[axis] % tile_size != 0:
                    raise Exception(
                        f"Outer tile is not divisible by inner tile on axis {axis}"
                    )

                if (
                    axis in current_split_size
                    and current_split_size[axis] % tile_size != 0
                ):  # Try to create tile that do not fit in the current split
                    raise Exception(
                        f"Current split on axis {axis} of size {current_split_size[axis]} cannot support tiles of size {tile_size}"
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
