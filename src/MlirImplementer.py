#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing_extensions import override
from mlir.ir import (
    InsertionPoint,
    UnitAttr,
)
from mlir.dialects import transform
from xdsl.dialects import func as xdslfunc
from mlir.dialects.transform import (
    structured,
    vector,
    memref,
    get_parent_op,
)

from MlirModule import MlirModule


class MlirImplementer(MlirModule, ABC):
    def __init__(
        self,
        xdsl_func: xdslfunc.FuncOp,
        always_vectorize: bool,
        no_alias: bool,
        concluding_passes: list[str],
    ):
        #
        super().__init__(xdsl_func, no_alias)
        #
        self.always_vectorize = always_vectorize
        self.concluding_passes = concluding_passes
        #

    @property
    def mlir_payload(self):
        return self.payload_name

    def generate_vectorization(self, handle):
        handle = get_parent_op(
            transform.AnyOpType.get(),
            handle,
            isolated_from_above=True,
        )
        if self.always_vectorize or self.needs_vectorization():
            handle = structured.VectorizeChildrenAndApplyPatternsOp(handle)
            # with InsertionPoint(transform.ApplyPatternsOp(handle).patterns):
            #     memref.ApplyFoldMemrefAliasOpsPatternsOp()
            # handle = structured.HoistRedundantVectorTransfersOp(target=handle,transformed=transform.AnyOpType.get())
            with InsertionPoint(transform.ApplyPatternsOp(handle).patterns):
                vector.ApplyLowerOuterProductPatternsOp()
                vector.ApplyLowerContractionPatternsOp()
        return handle

    @abstractmethod
    def needs_vectorization(self):
        pass

    @abstractmethod
    def check_consistency(self):
        pass

    @abstractmethod
    def generate_tiling(self):
        pass

    @abstractmethod
    def generate_unroll(self, handle):
        pass

    @override
    def _implement(self, measure=True):
        with (
            InsertionPoint.at_block_begin(self.named_sequence.body),
            self.mlir_context,
            self.loc,
        ):
            handle = self.generate_tiling()
            handle = self.generate_vectorization(handle)
            self.generate_unroll(handle)
            for p in self.concluding_passes:
                handle = transform.ApplyRegisteredPassOp(
                    transform.AnyOpType.get(), handle, pass_name=p
                )
            transform.YieldOp([])

    @abstractmethod
    def string_of_schedule(self):
        pass

    @abstractmethod
    def np_inputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        pass

    @abstractmethod
    def np_outputs_spec(self) -> list[dict[str, tuple[int, ...] | str]]:
        pass

    @abstractmethod
    def reference_impl(self, *operands):
        pass
