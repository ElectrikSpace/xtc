#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..MlirConfig import MlirConfig
from ..MlirProgram import RawMlirProgram

import xtc.itf as itf
from xtc.itf.graph import Graph

from mlir.ir import OpResult

__all__ = ["MlirTarget"]


class MlirTarget(ABC):
    """An abstract Mlir Target for the Mlir Compiler

    A Mlir Target is responsible for the final steps of the lowering and code generation
    for a specific Target. A Target can be the software backend dedicated to a Hardware,      or a new code generation for an already known Hardware.
    """

    def __init__(self, config: MlirConfig):
        self._config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_config: MlirConfig):
        self._config = new_config

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def arch(self) -> str: ...

    @abstractmethod
    def generate_code_for_target(
        self, mlir_program: RawMlirProgram, **kwargs: Any
    ) -> None:
        """
        Generate the code and produce the shared lib or executable for the target.
        The mlir_program represents the IR after XTC transformations.
        """
        ...

    @abstractmethod
    def create_module(
        self,
        name: str,
        payload_name: str,
        file_name: str,
        file_type: str,
        graph: Graph | None = None,
        **kwargs: Any,
    ) -> itf.comp.Module:
        """
        Create a Module instance that represents the result of a previous call to
        generate_code_for_target.
        It can return a derived class from Module that handle runtime specificities
        for the target.
        """
        ...

    @abstractmethod
    def has_custom_vectorize(self) -> bool:
        """
        Return True if the target needs to apply custom vectorization.
        """
        ...

    @abstractmethod
    def apply_custom_vectorize(self, handle: OpResult) -> None:
        """
        Apply the custom vectorization for the target.
        """
        ...

    def pack_at_padding_heuristic(
        self,
        *,
        schedule_dims: Sequence[str],
        input_idx: int,
        input_element_bytewidth: int,
        input_buffer_shape: Sequence[int],
    ) -> dict[str, int]:
        """Suggested per-dimension slack when ``pack_at(..., pad=True)``.

        Dimensions are abstract nest names (e.g. ``i``, ``j``, ``k`` for matmul).
        Returning only zeros disables heuristic padding for every target by default;
        specific targets (e.g. MPPA) may override with a TVM-style stride slack hint.

        Args:
            schedule_dims: Ordered dimension names for the scheduled operation.
            input_idx: Packed operand index.
            input_element_bytewidth: Element size in bytes for that operand.
            input_buffer_shape: Operand shape **after ancestor tiling** at the ``pack_at``
                loop (memref rank order), computed by the MLIR compiler from the schedule.
        """
        del schedule_dims, input_idx, input_element_bytewidth, input_buffer_shape
        return {}
