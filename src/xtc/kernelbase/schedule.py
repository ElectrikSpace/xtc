#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xtc.itf.schd import Scheduler
from xtc.schedules.descript import descript_scheduler

__all__ = [
    "DescriptSchedule",
]


@dataclass
class DescriptSchedule:
    """Serializable descript schedule (see descript_scheduler)."""

    node_name: str
    abstract_dims: list[str]
    spec: dict[str, Any] | str
    abstract_dim_sizes: dict[str, int] = field(default_factory=dict)
    abstract_matrix: list[str] = field(default_factory=list)
    sample: dict[str, int] = field(default_factory=dict)
    default_node: str | None = None

    def apply(self, scheduler: Scheduler) -> None:
        descript_scheduler(
            scheduler=scheduler,
            node_name=self.node_name,
            abstract_dims=self.abstract_dims,
            spec=self.spec,
            abstract_dim_sizes=self.abstract_dim_sizes,
            abstract_matrix=self.abstract_matrix,
            sample=self.sample,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_name": self.node_name,
            "abstract_dims": self.abstract_dims,
            "spec": self.spec,
        }
        if self.abstract_dim_sizes:
            d["abstract_dim_sizes"] = self.abstract_dim_sizes
        if self.abstract_matrix:
            d["abstract_matrix"] = self.abstract_matrix
        if self.sample:
            d["sample"] = self.sample
        if self.default_node is not None:
            d["default_node"] = self.default_node
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DescriptSchedule:
        return cls(
            node_name=data["node_name"],
            abstract_dims=list(data["abstract_dims"]),
            spec=data["spec"],
            abstract_dim_sizes=dict(data.get("abstract_dim_sizes", {})),
            abstract_matrix=list(data.get("abstract_matrix", [])),
            sample=dict(data.get("sample", {})),
            default_node=data.get("default_node"),
        )

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> DescriptSchedule:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        return cls.from_dict(data)
