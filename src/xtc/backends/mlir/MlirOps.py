#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing_extensions import override
from typing import Any, Type, TypeAlias, cast

from xdsl.dialects import linalg, arith, builtin, memref, tensor, scf, math
from xdsl.dialects.builtin import (
    MemRefType,
    TensorType,
    IndexType,
    f16,
    f32,
    f64,
    i8,
    i32,
    i64,
    UnitAttr,
    StringAttr,
    AffineMapAttr,
)
from xdsl.ir import Block, BlockArgument, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.irdl import irdl_op_definition
from xdsl.builder import ImplicitBuilder

from xtc.itf.graph import Operation
from xtc.graphs.xtc.data import XTCTensorType
from xtc.quant.ggml import GGMLField, ggml_quant_format
from xtc.utils.math import mulall


__all__ = [
    "MlirOperation",
    "MlirOperator",
    "MlirOperators",
]

OpAttrs: TypeAlias = dict[str, Any]


def _is_natural_layout(layout: list[int] | None, ndim: int) -> bool:
    return layout is None or layout == list(range(ndim))


def _require_natural_layouts(xtc_op: Operation) -> None:
    for typ in xtc_op.inputs_types:
        if isinstance(typ, XTCTensorType) and not _is_natural_layout(
            typ.layout, typ.ndim
        ):
            raise NotImplementedError(
                "tensor layout is not yet implemented in MLIR backend"
            )


class MlirOperation:
    def __init__(
        self,
        operator: Type["MlirOperator"],
        args: tuple[Any, ...],
        attrs: dict[str, Any] = {},
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
    ) -> None:
        self.operator = operator(args, attrs, name=name, op_type=op_type)
        self.args = args
        self.attrs = attrs
        self.name = self.operator.name if name is None else name

    def generate(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        return self.operator.generate_op(block, args)

    def np_inputs_spec(self) -> list[dict[str, Any]]:
        inputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.inputs_dims(), self.operator.inputs_types()
            )
        ]
        return inputs_spec

    def np_outputs_spec(self) -> list[dict[str, Any]]:
        outputs_spec = [
            {
                "shape": shape,
                "dtype": dtype,
            }
            for shape, dtype in zip(
                self.operator.outputs_dims(), self.operator.outputs_types()
            )
        ]
        return outputs_spec

    @classmethod
    def from_operation(
        cls,
        xtc_op: Operation,
        name: str | None,
        op_type: Type[MemRefType] | Type[TensorType],
    ) -> "MlirOperation":
        dims = xtc_op.dims.values()
        dtype = xtc_op.inputs_types[0].dtype  # TODO: currently get dtype from 1st arg
        args = tuple([*dims, dtype])
        attrs = xtc_op.attrs
        args = MlirOperators.from_name(xtc_op.name).args_from_operation(xtc_op, args)
        if xtc_op.name == "matmul":
            transpose_a = xtc_op.inputs_types[0].layout == [1, 0]
            transpose_b = xtc_op.inputs_types[1].layout == [1, 0]
            in_dtype = xtc_op.inputs_types[0].dtype
            acc_dtype = xtc_op.outputs_types[0].dtype
            args = tuple([*dims, in_dtype, acc_dtype, transpose_a, transpose_b])
            return MlirOperation(
                MlirOperators.from_name(xtc_op.name),
                args,
                dict(attrs),
                name=name,
                op_type=op_type,
            )
        return MlirOperation(
            MlirOperators.from_name(xtc_op.name),
            args,
            dict(attrs),
            name=name,
            op_type=op_type,
        )


class MlirOperator(ABC):
    DEFAULT_NAME = "undef"
    AXES = ""
    KINDS = ""

    def __init__(
        self,
        args: tuple[Any, ...],
        attrs: dict[str, Any],
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
    ) -> None:
        self.args = args
        self.attrs = {**attrs}
        self.name = name if name is not None else self.DEFAULT_NAME
        self.op_type = op_type

    @abstractmethod
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]: ...
    @abstractmethod
    def dims(self, kind: str = "") -> tuple[str, ...]: ...
    @abstractmethod
    def dims_sizes(self) -> dict[str, int]: ...
    @abstractmethod
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def inputs_types(self) -> tuple[str, ...]: ...
    @abstractmethod
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]: ...
    @abstractmethod
    def outputs_types(self) -> tuple[str, ...]: ...

    def _dims(self, kind: str = "") -> tuple[str, ...]:
        if kind == "":
            return tuple(self.AXES)
        return tuple([a for a, k in zip(self.AXES, self.KINDS) if k == kind])

    @classmethod
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        _require_natural_layouts(xtc_op)
        return args


class MlirOperatorMatmul(MlirOperator):
    DEFAULT_NAME = "matmul"
    AXES = "ijk"
    KINDS = "PPR"

    @classmethod
    @override
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        transpose_a = xtc_op.inputs_types[0].layout == [1, 0]
        transpose_b = xtc_op.inputs_types[1].layout == [1, 0]
        return (*args, transpose_a, transpose_b)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, j, k, *_ = self.args
        return {"i": i, "j": j, "k": k}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Ki, Kj, Kk, input_dtype, acc_dtype, transpose_a, transpose_b = self.args
        str_to_builtin = {"float16": f16, "float32": f32, "float64": f64}
        str_to_bits = {"float16": 16, "float32": 32, "float64": 64}
        in_elt = str_to_builtin[input_dtype]
        acc_elt = str_to_builtin[acc_dtype]
        acc_bits = str_to_bits[acc_dtype]
        if block is None:
            ops_types = [
                self.op_type(in_elt, [Ki, Kk]),
                self.op_type(in_elt, [Kk, Kj]),
                self.op_type(acc_elt, [Ki, Kj]),
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        use_linalg_matmul = (
            not transpose_a and not transpose_b and input_dtype == acc_dtype
        )
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, acc_bits))
            result = (args[2].type,) if self.op_type == TensorType else ()
            fill = linalg.FillOp(
                res=result,
                inputs=(cst0.results[0],),
                outputs=(args[2],),
            )
            if use_linalg_matmul:
                reduce = linalg.MatmulOp(
                    res=result,
                    inputs=(args[0], args[1]),
                    outputs=(fill.results[0],)
                    if self.op_type == TensorType
                    else (args[2],),
                )
            else:
                iterator_types = [
                    StringAttr("parallel"),
                    StringAttr("parallel"),
                    StringAttr("reduction"),
                ]
                index_map_a = lambda i, j, k: (i, k)
                index_map_b = lambda i, j, k: (k, j)
                index_map_c = lambda i, j, k: (i, j)
                if transpose_a:
                    index_map_a = lambda i, j, k: (k, i)
                if transpose_b:
                    index_map_b = lambda i, j, k: (j, k)
                elt_type = {"float16": f16, "float32": f32, "float64": f64}[input_dtype]
                block_in = Block(arg_types=[in_elt, in_elt, acc_elt])
                with ImplicitBuilder(block_in):
                    if input_dtype == acc_dtype:
                        mul = arith.MulfOp(block_in.args[0], block_in.args[1])
                    else:
                        ext_a = arith.ExtFOp(block_in.args[0], acc_elt)
                        ext_b = arith.ExtFOp(block_in.args[1], acc_elt)
                        mul = arith.MulfOp(ext_a, ext_b)
                    add = arith.AddfOp(block_in.args[2], mul)
                    linalg.YieldOp(add)
                reduce = linalg.GenericOp(
                    inputs=(args[0], args[1]),
                    outputs=(fill.results[0],)
                    if self.op_type == TensorType
                    else (args[2],),
                    body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                    # ignore typing due to xdsl hints limitation
                    indexing_maps=[
                        AffineMapAttr(AffineMap.from_callable(index_map_a)),
                        AffineMapAttr(AffineMap.from_callable(index_map_b)),
                        AffineMapAttr(AffineMap.from_callable(index_map_c)),
                    ],
                    iterator_types=iterator_types,
                    result_types=result,
                )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"i": Ki, "j": Kj},
                self.dims_sizes(),
            ],
            "reduction_dims": [[], list(self.dims("R"))],
            "output_nodes": [reduce],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j, k, *_ = self.args
        return (i, k), (k, j)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        input_dtype = self.args[3]
        return input_dtype, input_dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i, j = self.args[:2]
        return ((i, j),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        acc_dtype = self.args[4]
        return (acc_dtype,)


@irdl_op_definition
class Conv2DNhwcHwFcOp(linalg.ConvOperation):
    """
    Performs 2-D convolution with inputs (N, H, W, C) (R, S, C F)

    See https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nhwc_hwcf-linalgconv2dnhwchwcfop
    """

    name = "linalg.conv_2d_nhwc_hwcf"


class MlirOperatorConv2D(MlirOperator):
    DEFAULT_NAME = "conv2d"
    AXES = "bhwfrsc"
    KINDS = "PPPPRRR"

    DEFAULT_STRIDE = (1, 1)

    def __init__(
        self,
        args: tuple[Any, ...],
        attrs: dict[str, Any],
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
    ) -> None:
        attrs = {"stride": self.DEFAULT_STRIDE, **attrs}
        super().__init__(args, attrs, name, op_type)

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        b, h, w, f, r, s, c, _ = self.args
        return {"b": b, "h": h, "w": w, "f": f, "r": r, "s": s, "c": c}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Kb, Kh, Kw, Kf, Kr, Ks, Kc, dtype = self.args
        SH, SW = self.attrs["stride"]
        inps_dims = self.inputs_dims()
        out_dims = self.outputs_dims()[0]
        dtype = self.args[-1]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape) for shape in [*inps_dims, out_dims]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        with ImplicitBuilder(block):
            result = (args[2].type,) if self.op_type == TensorType else ()
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            fill = linalg.FillOp(
                res=result,
                inputs=(cst0.results[0],),
                outputs=(args[2],),
            )
            # TODO: Does not work
            # strides = DenseIntOrFPElementsAttr.vector_from_list([SH, SW], i64)
            # dilations = DenseIntOrFPElementsAttr.vector_from_list([1, 1], i64)
            # reduce = Conv2DNhwcHwFcOp(
            #     inputs=(block.args[0], block.args[1]),
            #     outputs=(block.args[2],),
            #     dilations=dilations,
            #     strides=strides,
            # )
            iterator_types = [
                StringAttr({"P": "parallel", "R": "reduction"}[k]) for k in self.KINDS
            ]
            flags = arith.FastMathFlagsAttr("fast")
            block_in = Block(arg_types=[f32, f32, f32])
            with ImplicitBuilder(block_in):
                mul = arith.MulfOp(block_in.args[0], block_in.args[1], flags=flags)
                add = arith.AddfOp(block_in.args[2], mul, flags=flags)
                linalg.YieldOp(add)
            reduce = linalg.GenericOp(
                inputs=(args[0], args[1]),
                outputs=(fill.results[0],)
                if self.op_type == TensorType
                else (args[2],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                # ignore typing due to xdsl hints limitation
                indexing_maps=[
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (b, h * SH + r, w * SW + s, c)
                            )
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (r, s, c, f)
                            )
                        )
                    ),
                    AffineMapAttr(
                        AffineMap.from_callable(
                            lambda b, h, w, f, r, s, c: (  # type: ignore
                                (b, h, w, f)
                            )
                        )
                    ),
                ],
                iterator_types=iterator_types,
                result_types=result,
            )
        fill_node_id = f"{self.name}_0"
        reduce_node_id = f"{self.name}"
        fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        reduce.attributes[f"__xtc_id_{reduce_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                fill_node_id: fill,
                reduce_node_id: reduce,
            },
            "dims_sizes": [
                {"b": Kb, "h": Kh, "w": Kw, "f": Kf},
                self.dims_sizes(),
            ],
            "reduction_dims": [[], list(self.dims("R"))],
            "output_nodes": [reduce],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f, r, s, c, _ = self.args
        SH, SW = self.attrs["stride"]
        return ((b, h * SH + r - 1, w * SW + s - 1, c), (r, s, c, f))

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return dtype, dtype

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        b, h, w, f = self.args[:4]
        return ((b, h, w, f),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorRelu(MlirOperator):
    DEFAULT_NAME = "relu"
    AXES = "i"
    KINDS = "P"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        i, _ = self.args
        return {"i": i}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        Ki, dtype = self.args
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [self.op_type(elt_type, shape) for shape in [[Ki], [Ki]]]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        inp_shape, out_shape = [
            list(cast(self.op_type, arg.type).get_shape())  # type: ignore
            for arg in args
        ]
        inp_size, out_size = [mulall(shape) for shape in [inp_shape, out_shape]]
        assert inp_size == out_size
        with ImplicitBuilder(block):
            inp_reassociation = builtin.ArrayAttr(
                [
                    builtin.ArrayAttr(
                        [builtin.IntegerAttr(x, i64) for x in range(len(inp_shape))]
                    )
                ]
            )
            out_reassociation = builtin.ArrayAttr(
                [
                    builtin.ArrayAttr(
                        [builtin.IntegerAttr(x, i64) for x in range(len(out_shape))]
                    )
                ]
            )
            if self.op_type == TensorType:
                # TODO: re-introduce collapsing relu
                out_operand = args[1]
                inp_operand = args[0]
                rank = len(out_shape)
                iterator_types = [StringAttr("parallel")] * rank
                indexing_maps = [
                    AffineMapAttr(AffineMap.identity(rank)),  # input
                    AffineMapAttr(
                        AffineMap.identity(rank).drop_results(out_shape)
                    ),  # scalar
                    AffineMapAttr(AffineMap.identity(rank)),  # output
                ]
            else:
                inp = memref.CollapseShapeOp(  # type: ignore
                    operands=[args[0]],
                    properties=dict(reassociation=inp_reassociation),
                    result_types=[self.op_type(elt_type, (inp_size,))],
                )
                inp_operand = inp.results[0]  # type: ignore
                out = memref.CollapseShapeOp(
                    operands=[args[1]],
                    properties=dict(reassociation=out_reassociation),
                    result_types=[self.op_type(elt_type, (out_size,))],
                )
                out_operand = out.results[0]  # type: ignore
                iterator_types = [
                    StringAttr({"P": "parallel", "R": "reduction"}[k])
                    for k in self.KINDS
                ]
                # ignore typing due to xdsl hints limitation
                indexing_maps = [
                    AffineMapAttr(AffineMap.from_callable(lambda i: (i,))),  # type: ignore
                    AffineMapAttr(AffineMap.from_callable(lambda _: ())),  # type: ignore
                    AffineMapAttr(AffineMap.from_callable(lambda i: (i,))),  # type: ignore
                ]
                iterator_types = [
                    StringAttr({"P": "parallel", "R": "reduction"}[k])
                    for k in self.KINDS
                ]
            result = (args[1].type,) if self.op_type == TensorType else ()
            cst0 = arith.ConstantOp(builtin.FloatAttr(0, elt_size))
            block_in = Block(arg_types=[f32, f32, f32])
            with ImplicitBuilder(block_in):
                max = arith.MaximumfOp(block_in.args[0], block_in.args[1])
                linalg.YieldOp(max)
            relu = linalg.GenericOp(
                inputs=(inp_operand, cst0.results[0]),
                outputs=(out_operand,),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
                result_types=result,
            )
        relu_node_id = f"{self.name}"
        relu.attributes[f"__xtc_id_{relu_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                relu_node_id: relu,
            },
            "dims_sizes": [
                self.dims_sizes(),
            ],
            "output_nodes": [relu],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i = self.args[0]
        return ((i,),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        i = self.args[0]
        return ((i,),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorPad(MlirOperator):
    DEFAULT_NAME = "pad"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        assert len(self.args[:-1]) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        dtype = self.args[-1]
        dims_value = list(self.args[:-1])
        padding = self.attrs["padding"]
        constant_value = self.attrs["constant_value"]
        lows = [0] * len(dims_value)
        highs = [0] * len(dims_value)
        if isinstance(padding, dict):
            dims_value_before_pad = list(dims_value)
            for i, pad_value in padding.items():
                dims_value_before_pad[i] -= sum(pad_value)
                lows[i] = pad_value[0]
                highs[i] = pad_value[1]
        else:
            dims_value_before_pad = [
                dim_value - sum(padding) for dim_value in dims_value
            ]
            lows = [padding[0] for d in dims_value]
            highs = [padding[1] for d in dims_value]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        elt_size = {"float32": 32, "float64": 64}[dtype]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for shape in [dims_value_before_pad, dims_value]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args[:-1]]
            for i, (pad_b, pad_a) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args[:-1]]
        sizes = list(dims_value_before_pad)
        strides = [1 for _ in self.args[:-1]]
        using_tensors = self.op_type == TensorType
        with ImplicitBuilder(block):
            cst0 = arith.ConstantOp(builtin.FloatAttr(constant_value, elt_size))
            result = (args[1].type,) if using_tensors else ()
            fill_node_id = f"{self.name}_0"

            if using_tensors:
                fill = None
                empty = args[1]
                block_in = Block(arg_types=[elt_type])
                rank = len(dims_value)
                # pad written as a linalg.generic to enable producer fusion
                with ImplicitBuilder(block_in):
                    # gets the current iteration index for each dim (not constants)
                    output_indices = [linalg.IndexOp(i) for i in range(rank)]
                    input_indices = []
                    in_bounds_checks = []

                    zero = arith.ConstantOp.create(
                        properties={"value": builtin.IntegerAttr(0, IndexType())},
                        result_types=[IndexType()],
                    )

                    for dim_idx in range(rank):
                        lo_const = arith.ConstantOp.create(
                            properties={
                                "value": builtin.IntegerAttr(lows[dim_idx], IndexType())
                            },
                            result_types=[IndexType()],
                        )
                        input_idx = arith.SubiOp(output_indices[dim_idx], lo_const)
                        input_indices.append(input_idx)

                        # check to see if in the padding region or input tensor region
                        size_const = arith.ConstantOp.create(
                            properties={
                                "value": builtin.IntegerAttr(
                                    dims_value_before_pad[dim_idx], IndexType()
                                )
                            },
                            result_types=[IndexType()],
                        )

                        ge_zero = arith.CmpiOp(input_idx, zero, "sge")
                        lt_size = arith.CmpiOp(input_idx, size_const, "slt")

                        in_bounds_checks.append(ge_zero)
                        in_bounds_checks.append(lt_size)

                    all_in_bounds: arith.CmpiOp | arith.AndIOp = in_bounds_checks[0]
                    for check in in_bounds_checks[1:]:
                        all_in_bounds = arith.AndIOp(all_in_bounds, check)

                    if_region_then = Region([Block(arg_types=[])])
                    if_region_else = Region([Block(arg_types=[])])

                    with ImplicitBuilder(if_region_then.blocks[0]):
                        extracted = tensor.ExtractOp(
                            tensor=args[0],
                            indices=input_indices,  # type: ignore
                            result_type=elt_type,
                        )
                        scf.YieldOp(extracted)
                    with ImplicitBuilder(if_region_else.blocks[0]):
                        scf.YieldOp(cst0)

                    if_op = scf.IfOp(
                        cond=all_in_bounds,
                        true_region=if_region_then,
                        false_region=if_region_else,
                        return_types=[elt_type],
                    )

                    linalg.YieldOp(if_op.results[0])

                copy = linalg.GenericOp(
                    inputs=[],
                    outputs=[empty],
                    body=Region([block_in]),
                    indexing_maps=[AffineMapAttr(AffineMap.identity(rank))],
                    iterator_types=[StringAttr("parallel")] * rank,
                    result_types=[TensorType(elt_type, dims_value)],
                )
            else:
                fill = linalg.FillOp(
                    res=result,
                    inputs=(cst0.results[0],),
                    outputs=(args[1],),
                )
                subview = memref.SubviewOp.from_static_parameters(
                    source=args[1],
                    source_type=args[1].type,  # type: ignore
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                )
                copy = linalg.CopyOp(  # type: ignore
                    inputs=[args[0]],
                    outputs=[subview.result],
                    res=result,
                )
                fill.attributes[f"__xtc_id_{fill_node_id}_"] = UnitAttr()
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                **({fill_node_id: fill} if fill else {}),
                copy_node_id: copy,
            },
            "dims_sizes": [
                self.dims_sizes(),
                *([] if using_tensors else [self.dims_sizes()]),
            ],
            "output_nodes": [copy],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        dims_value = list(self.args[:-1])
        if isinstance(padding, dict):
            for i, pad_value in padding.items():
                dims_value[i] -= sum(pad_value)
        else:
            dims_value = [value - sum(padding) for value in dims_value]
        return (tuple(dims_value),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorEltwiseBinary(MlirOperator):
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"
    ARITH_OP: Any = None

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        rank = len(self.args) - 1
        return self._dims(kind)[:rank]

    @override
    def dims_sizes(self) -> dict[str, int]:
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        shape = list(self.args[:-1])
        dtype = self.args[-1]
        rank = len(shape)
        elt_type = {"float16": f16, "float32": f32, "float64": f64}[dtype]
        if block is None:
            ops_types = [self.op_type(elt_type, shape) for _ in range(3)]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 3
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        with ImplicitBuilder(block):
            ident = AffineMapAttr(AffineMap.identity(rank))
            indexing_maps = [ident, ident, ident]
            iterator_types = [StringAttr("parallel")] * rank
            result = (args[2].type,) if self.op_type == TensorType else ()
            block_in = Block(arg_types=[elt_type, elt_type, elt_type])
            with ImplicitBuilder(block_in):
                res = self.ARITH_OP(block_in.args[0], block_in.args[1])
                linalg.YieldOp(res)
            node = linalg.GenericOp(
                inputs=(args[0], args[1]),
                outputs=(args[2],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
                result_types=result,
            )
        node_id = f"{self.name}"
        node.attributes[f"__xtc_id_{node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {node_id: node},
            "dims_sizes": [self.dims_sizes()],
            "output_nodes": [node],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]), tuple(self.args[:-1]))

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype, dtype)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


class MlirOperatorAdd(MlirOperatorEltwiseBinary):
    DEFAULT_NAME = "add"
    ARITH_OP = arith.AddfOp


class MlirOperatorMul(MlirOperatorEltwiseBinary):
    DEFAULT_NAME = "mul"
    ARITH_OP = arith.MulfOp


class MlirOperatorFma(MlirOperator):
    DEFAULT_NAME = "fma"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)[: len(self.args) - 1]

    @override
    def dims_sizes(self) -> dict[str, int]:
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        shape = list(self.args[:-1])
        dtype = self.args[-1]
        rank = len(shape)
        elt_type = {"float16": f16, "float32": f32, "float64": f64}[dtype]
        if block is None:
            ops_types = [self.op_type(elt_type, shape) for _ in range(4)]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 4
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        with ImplicitBuilder(block):
            ident = AffineMapAttr(AffineMap.identity(rank))
            result = (args[3].type,) if self.op_type == TensorType else ()
            block_in = Block(arg_types=[elt_type] * 4)
            with ImplicitBuilder(block_in):
                mul = arith.MulfOp(block_in.args[0], block_in.args[1])
                add = arith.AddfOp(mul, block_in.args[2])
                linalg.YieldOp(add)
            node = linalg.GenericOp(
                inputs=(args[0], args[1], args[2]),
                outputs=(args[3],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=[ident] * 4,
                iterator_types=[StringAttr("parallel")] * rank,
                result_types=result,
            )
        node_id = f"{self.name}"
        node.attributes[f"__xtc_id_{node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {node_id: node},
            "dims_sizes": [self.dims_sizes()],
            "output_nodes": [node],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]),) * 3

    @override
    def inputs_types(self) -> tuple[str, ...]:
        return (self.args[-1],) * 3

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:-1]),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        return (self.args[-1],)


class MlirOperatorEltwiseUnary(MlirOperator):
    """Elementwise unary function; the body is given by the ``body`` method."""

    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"
    ELT_TYPES = {"float16": f16, "float32": f32, "float64": f64}

    def _in_dtype(self) -> str:
        return self.args[-1]

    def _out_dtype(self) -> str:
        return self.args[-1]

    def _shape(self) -> list[int]:
        return list(self.args[:-1])

    @staticmethod
    def body(x: SSAValue, elt_in: Any, elt_out: Any) -> SSAValue:
        raise NotImplementedError

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)[: len(self._shape())]

    @override
    def dims_sizes(self) -> dict[str, int]:
        return {name: size for name, size in zip(self.AXES, self._shape())}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        shape = self._shape()
        rank = len(shape)
        elt_in = self.ELT_TYPES[self._in_dtype()]
        elt_out = self.ELT_TYPES[self._out_dtype()]
        if block is None:
            block = Block(
                arg_types=[self.op_type(elt_in, shape), self.op_type(elt_out, shape)]
            )
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        with ImplicitBuilder(block):
            ident = AffineMapAttr(AffineMap.identity(rank))
            result = (args[1].type,) if self.op_type == TensorType else ()
            block_in = Block(arg_types=[elt_in, elt_out])
            with ImplicitBuilder(block_in):
                res = self.body(block_in.args[0], elt_in, elt_out)
                linalg.YieldOp(res)
            node = linalg.GenericOp(
                inputs=(args[0],),
                outputs=(args[1],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=[ident, ident],
                iterator_types=[StringAttr("parallel")] * rank,
                result_types=result,
            )
        node_id = f"{self.name}"
        node.attributes[f"__xtc_id_{node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {node_id: node},
            "dims_sizes": [self.dims_sizes()],
            "output_nodes": [node],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self._shape()),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        return (self._in_dtype(),)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self._shape()),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        return (self._out_dtype(),)


class MlirOperatorExp(MlirOperatorEltwiseUnary):
    DEFAULT_NAME = "exp"

    @staticmethod
    @override
    def body(x: SSAValue, elt_in: Any, elt_out: Any) -> SSAValue:
        return math.ExpOp(x).result


class MlirOperatorRsqrt(MlirOperatorEltwiseUnary):
    DEFAULT_NAME = "rsqrt"

    @staticmethod
    @override
    def body(x: SSAValue, elt_in: Any, elt_out: Any) -> SSAValue:
        return math.RsqrtOp(x).result


class MlirOperatorSilu(MlirOperatorEltwiseUnary):
    DEFAULT_NAME = "silu"

    @staticmethod
    @override
    def body(x: SSAValue, elt_in: Any, elt_out: Any) -> SSAValue:
        neg = arith.NegfOp(x)
        ex = math.ExpOp(neg.result)
        one = arith.ConstantOp(builtin.FloatAttr(1.0, elt_in))
        den = arith.AddfOp(one.result, ex.result)
        return arith.DivfOp(x, den.result).result


class MlirOperatorCast(MlirOperatorEltwiseUnary):
    DEFAULT_NAME = "cast"

    @classmethod
    @override
    def args_from_operation(
        cls, xtc_op: Operation, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        _require_natural_layouts(xtc_op)
        return (*args, xtc_op.outputs_types[0].dtype)

    @override
    def _out_dtype(self) -> str:
        return self.args[-1]

    @override
    def _in_dtype(self) -> str:
        return self.args[-2]

    @override
    def _shape(self) -> list[int]:
        return list(self.args[:-2])

    @staticmethod
    @override
    def body(x: SSAValue, elt_in: Any, elt_out: Any) -> SSAValue:
        if elt_in == elt_out:
            return x
        bits = {f16: 16, f32: 32, f64: 64}
        op = arith.TruncFOp if bits[elt_out] < bits[elt_in] else arith.ExtFOp
        return op(x, elt_out).result


class MlirOperatorPad2D(MlirOperatorPad):
    DEFAULT_NAME = "pad2d"
    AXES = "bhwc"
    KINDS = "PPPP"


class MlirOperatorUnpad(MlirOperator):
    DEFAULT_NAME = "unpad"
    AXES = "ijklmnopqrstuvwxyz"
    KINDS = "PPPPPPPPPPPPPPPPPP"

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        return self._dims(kind)

    @override
    def dims_sizes(self) -> dict[str, int]:
        assert len(self.args[:-1]) <= len(self.AXES)
        return {name: size for name, size in zip(self.AXES, self.args[:-1])}

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        dtype = self.args[-1]
        dims_values = list(self.args[:-1])
        padding = self.attrs["padding"]
        if isinstance(padding, dict):
            dims_values_before_unpad = list(dims_values)
            for i, pad_value in padding.items():
                dims_values_before_unpad[i] += sum(pad_value)
        else:
            dims_values_before_unpad = [
                dim_value + sum(padding) for dim_value in dims_values
            ]
        elt_type = {"float32": f32, "float64": f64}[dtype]
        if block is None:
            ops_types = [
                self.op_type(elt_type, shape)
                for shape in [dims_values_before_unpad, dims_values]
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        assert all(isinstance(arg.type, self.op_type) for arg in args)
        if isinstance(padding, dict):
            offsets = [0 for _ in self.args[:-1]]
            for i, (pad_b, _) in padding.items():
                offsets[i] = pad_b
        else:
            offsets = [padding[0] for _ in self.args[:-1]]
        sizes = dims_values
        strides = [1 for _ in self.args[:-1]]
        using_tensors = self.op_type == TensorType
        with ImplicitBuilder(block):
            if using_tensors:
                copy = tensor.ExtractSliceOp.from_static_parameters(
                    source=args[0],
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                )
            else:
                subview = memref.SubviewOp.from_static_parameters(
                    source=args[0],
                    source_type=args[0].type,  # type: ignore
                    offsets=offsets,
                    sizes=sizes,
                    strides=strides,
                )
                copy = linalg.CopyOp(  # type: ignore
                    inputs=[subview.result],
                    outputs=[args[1]],
                    res=(),
                )
        copy_node_id = f"{self.name}"
        copy.attributes[f"__xtc_id_{copy_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                copy_node_id: None if using_tensors else copy,
            },
            "dims_sizes": [*([] if using_tensors else [self.dims_sizes()])],
            "output_nodes": [copy],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        padding = self.attrs["padding"]
        inp_dims = list(self.args[:-1])
        if isinstance(padding, dict):
            for axis, (pad_b, pad_a) in padding.items():
                inp_dims[axis] += pad_b + pad_a
        else:
            inp_dims = [inp_dim + sum(padding) for inp_dim in inp_dims]
        return (tuple(inp_dims),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (self.args[:-1],)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        dtype = self.args[-1]
        return (dtype,)


def _fp16_bits_to_f32(lo: SSAValue, hi: SSAValue) -> SSAValue:
    """f32 value of the little endian fp16 bytes ``lo``, ``hi`` (i32 values).

    Decoded with integer arithmetic, as fp16 is not supported by all targets.
    Inf/NaN are not handled: ggml scales are finite.
    """
    c = lambda v: arith.ConstantOp(builtin.IntegerAttr(v, i32)).result
    h = arith.OrIOp(lo, arith.ShLIOp(hi, c(8))).result
    e = arith.AndIOp(arith.ShRUIOp(h, c(10)), c(0x1F)).result
    m = arith.AndIOp(h, c(0x3FF)).result
    normal_bits = arith.OrIOp(
        arith.ShLIOp(arith.AddiOp(e, c(112)), c(23)), arith.ShLIOp(m, c(13))
    )
    normal = arith.BitcastOp(normal_bits, f32).result
    subnormal = arith.MulfOp(
        arith.UIToFPOp(m, f32), arith.ConstantOp(builtin.FloatAttr(2.0**-24, f32))
    ).result
    mag = arith.SelectOp(arith.CmpiOp(e, c(0), "eq"), subnormal, normal).result
    is_neg = arith.CmpiOp(arith.AndIOp(h, c(0x8000)), c(0), "ne")
    return arith.SelectOp(is_neg, arith.NegfOp(mag), mag).result


class _MlirInt:
    """i32 value with the integer operators of ``GGMLDecodeOps`` (unsigned)."""

    def __init__(self, value: SSAValue) -> None:
        self.value = value

    @staticmethod
    def _ssa(x: "_MlirInt | int") -> SSAValue:
        if isinstance(x, _MlirInt):
            return x.value
        return arith.ConstantOp(builtin.IntegerAttr(x, i32)).result

    def _op(self, op: Any, other: "_MlirInt | int", rhs: bool = False) -> "_MlirInt":
        lhs, rhs_ = (other, self) if rhs else (self, other)
        return _MlirInt(op(self._ssa(lhs), self._ssa(rhs_)).result)

    __add__ = lambda self, o: self._op(arith.AddiOp, o)
    __radd__ = lambda self, o: self._op(arith.AddiOp, o, rhs=True)
    __sub__ = lambda self, o: self._op(arith.SubiOp, o)
    __rsub__ = lambda self, o: self._op(arith.SubiOp, o, rhs=True)
    __mul__ = lambda self, o: self._op(arith.MuliOp, o)
    __rmul__ = lambda self, o: self._op(arith.MuliOp, o, rhs=True)
    __and__ = lambda self, o: self._op(arith.AndIOp, o)
    __or__ = lambda self, o: self._op(arith.OrIOp, o)
    __lshift__ = lambda self, o: self._op(arith.ShLIOp, o)
    __rshift__ = lambda self, o: self._op(arith.ShRUIOp, o)
    __floordiv__ = lambda self, o: self._op(arith.DivUIOp, o)
    __mod__ = lambda self, o: self._op(arith.RemUIOp, o)


class _MlirFloat:
    """f32 value with the float operators of ``GGMLDecodeOps``."""

    def __init__(self, value: SSAValue) -> None:
        self.value = value

    __add__ = lambda self, o: _MlirFloat(arith.AddfOp(self.value, o.value).result)
    __sub__ = lambda self, o: _MlirFloat(arith.SubfOp(self.value, o.value).result)
    __mul__ = lambda self, o: _MlirFloat(arith.MulfOp(self.value, o.value).result)


class _MlirDecodeOps:
    @staticmethod
    def f32(x: _MlirInt) -> _MlirFloat:
        return _MlirFloat(arith.SIToFPOp(x.value, f32).result)

    @staticmethod
    def fp16(lo: _MlirInt, hi: _MlirInt) -> _MlirFloat:
        return _MlirFloat(_fp16_bits_to_f32(lo.value, hi.value))


class MlirOperatorDequantize(MlirOperator):
    """Dequantization of ggml blocks, see ``xtc.quant.ggml``.

    The quantized input (rows, K // qk, block_bytes) is read through one view
    per field of the format (e.g. scale low byte, scale high byte, quants)
    such that each element of the output only depends on affine accesses:
    tiling the op along k by multiples or divisors of qk gives slices of
    whole blocks, which a dequantizing pack can copy to local memory as is.
    """

    DEFAULT_NAME = "dequantize"
    KINDS = "PP"

    def __init__(
        self,
        args: tuple[Any, ...],
        attrs: dict[str, Any],
        name: str | None = None,
        op_type: Type[MemRefType] | Type[TensorType] = MemRefType,
    ) -> None:
        attrs = {"transpose": False, **attrs}
        super().__init__(args, attrs, name, op_type)
        self._fmt = ggml_quant_format(self.attrs["format"])

    @override
    def dims(self, kind: str = "") -> tuple[str, ...]:
        if kind == "R":
            return ()
        return ("k", "n") if self.attrs["transpose"] else ("n", "k")

    @override
    def dims_sizes(self) -> dict[str, int]:
        return dict(zip(self.dims(), self.args[:2]))

    @override
    def generate_op(
        self, block: Block | None = None, args: Sequence[BlockArgument] = []
    ) -> tuple[Block, OpAttrs]:
        fmt = self._fmt
        if block is None:
            ops_types = [
                self.op_type(i8, self.inputs_dims()[0]),
                self.op_type(f32, self.outputs_dims()[0]),
            ]
            block = Block(arg_types=ops_types)
            args = block.args
        assert len(args) == 2
        rows, nb, _ = self.inputs_dims()[0]
        k_idx = self.dims().index("k")
        if self.attrs["transpose"]:
            to_kn = lambda d0, d1: (d0, d1)
        else:
            to_kn = lambda d0, d1: (d1, d0)

        def field_map(field: GGMLField) -> AffineMap:
            def access(d0: Any, d1: Any) -> tuple[Any, ...]:
                k, n = to_kn(d0, d1)
                return (n, k // fmt.qk, field.index(k % fmt.qk))

            return AffineMap.from_callable(access)

        using_tensors = self.op_type == TensorType
        with ImplicitBuilder(block):
            views = []
            for field in fmt.fields:
                offsets = [0, 0, field.offset]
                sizes = [rows, nb, field.size]
                if using_tensors:
                    view = tensor.ExtractSliceOp.from_static_parameters(
                        source=args[0], offsets=offsets, sizes=sizes, strides=[1] * 3
                    )
                else:
                    view = memref.SubviewOp.from_static_parameters(
                        source=args[0],
                        source_type=args[0].type,  # type: ignore
                        offsets=offsets,
                        sizes=sizes,
                        strides=[1] * 3,
                    )
                views.append(view.result)
            block_in = Block(arg_types=[i8] * len(fmt.fields) + [f32])
            with ImplicitBuilder(block_in):
                k = arith.IndexCastOp(linalg.IndexOp(k_idx), i32).result
                e = _MlirInt(k) % fmt.qk
                fields = [
                    _MlirInt(arith.ExtUIOp(byte, i32).result)
                    for byte in block_in.args[:-1]
                ]
                value = fmt.decode(_MlirDecodeOps, e, *fields)
                linalg.YieldOp(value.value)
            dequant = linalg.GenericOp(
                inputs=views,
                outputs=(args[1],),
                body=Region([block_in]),  # type: ignore # mypy issue with dataclass
                indexing_maps=[
                    *(AffineMapAttr(field_map(field)) for field in fmt.fields),
                    AffineMapAttr(AffineMap.identity(2)),
                ],
                iterator_types=[StringAttr("parallel")] * 2,
                result_types=(args[1].type,) if using_tensors else (),
            )
        dequant_node_id = f"{self.name}"
        dequant.attributes[f"__xtc_id_{dequant_node_id}_"] = UnitAttr()
        attrs = {
            "nodes_map": {
                dequant_node_id: dequant,
            },
            "dims_sizes": [
                self.dims_sizes(),
            ],
            "output_nodes": [dequant],
        }
        return block, attrs

    @override
    def inputs_dims(self) -> tuple[tuple[int, ...], ...]:
        sizes = self.dims_sizes()
        return (self._fmt.quantized_shape(sizes["n"], sizes["k"]),)

    @override
    def inputs_types(self) -> tuple[str, ...]:
        return ("uint8",)

    @override
    def outputs_dims(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self.args[:2]),)

    @override
    def outputs_types(self) -> tuple[str, ...]:
        return ("float32",)


class MlirOperators:
    @classmethod
    def from_name(cls, name: str) -> Type[MlirOperator]:
        assert hasattr(cls, name), f"unknown operator name: {name}"
        return getattr(cls, name)

    matmul = MlirOperatorMatmul
    dequantize = MlirOperatorDequantize
    conv2d = MlirOperatorConv2D
    relu = MlirOperatorRelu
    add = MlirOperatorAdd
    mul = MlirOperatorMul
    fma = MlirOperatorFma
    exp = MlirOperatorExp
    rsqrt = MlirOperatorRsqrt
    silu = MlirOperatorSilu
    cast = MlirOperatorCast
    pad2d = MlirOperatorPad2D
    unpad = MlirOperatorUnpad
    pad = MlirOperatorPad
