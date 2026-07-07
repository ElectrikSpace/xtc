#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from __future__ import annotations

from typing import Any

from xtc.graphs.xtc.expr import XTCOpExpr, XTCTensorExpr
from xtc.graphs.xtc.graph import XTCGraph
from xtc.graphs.xtc.operators import XTCOperMatmul
from xtc.graphs.xtc.utils import XTCGraphUtils
from xtc.itf.graph import Graph
from xtc.utils.math import mulall

__all__ = ["generate_runtime_reference_cpp"]


def _matmul_dims(
    a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> tuple[int, int, int]:
    i, k = a_shape[0], mulall(list(a_shape[1:]))
    bk, j = mulall(list(b_shape[:-1])), b_shape[-1]
    assert k == bk, f"incompatible matmul shapes: {a_shape} x {b_shape}"
    return i, k, j


def generate_runtime_reference_cpp(
    graph: Graph,
    name_to_c_name: dict[str, str],
    name_to_shape: dict[str, tuple[int, ...]],
    output_c_names: list[str],
    c_type: str,
) -> tuple[str, str, list[str]]:
    """Generate C++ helpers and reference body for runtime validation.

    Returns:
        helper_functions, reference_body, output_ref_vars
    """
    if not isinstance(graph, XTCGraph):
        raise NotImplementedError(
            "runtime_validate export requires an XTC graph with codegen support"
        )

    helpers = f"""\
template <typename T>
void fill_random_inputs(std::vector<T>& data, uint32_t seed) {{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(1), static_cast<T>(9));
  for (auto& value : data) {{
    value = dist(rng);
  }}
}}

template <typename T>
void reference_matmul(const std::vector<T>& a, const std::vector<T>& b,
                      std::vector<T>& c, size_t i, size_t k, size_t j) {{
  std::fill(c.begin(), c.end(), static_cast<T>(0));
  for (size_t row = 0; row < i; ++row) {{
    for (size_t col = 0; col < j; ++col) {{
      T sum = static_cast<T>(0);
      for (size_t kk = 0; kk < k; ++kk) {{
        sum += a[row * k + kk] * b[kk * j + col];
      }}
      c[row * j + col] = sum;
    }}
  }}
}}
"""

    nodes = list(graph.nodes.values())
    topo = XTCGraphUtils.get_nodes_topological(nodes)
    uid_to_node: dict[str, Any] = {node.uid: node for node in graph.inputs_nodes}
    uid_to_node.update({node.uid: node for node in nodes})
    value_vars: dict[str, str] = {}
    for node in graph.inputs_nodes:
        value_vars[node.uid] = name_to_c_name[node.name]

    body_lines: list[str] = []
    output_ref_vars: list[str] = []

    for node in topo:
        if not isinstance(node._expr, XTCOpExpr):
            assert isinstance(node._expr, XTCTensorExpr)
            continue
        op = node.operator
        if isinstance(op, XTCOperMatmul):
            assert len(node.inputs) == 2
            a_uid, b_uid = node.inputs
            a_var = value_vars[a_uid]
            b_var = value_vars[b_uid]
            a_node = uid_to_node[a_uid]
            b_node = uid_to_node[b_uid]
            a_shape = name_to_shape[a_node.name]
            b_shape = name_to_shape[b_node.name]
            i_dim, k_dim, j_dim = _matmul_dims(a_shape, b_shape)
            out_c_name = name_to_c_name.get(node.name, output_c_names[0])
            if node.name in {n.name for n in graph.outputs_nodes}:
                ref_var = f"ref_{out_c_name}"
            else:
                ref_var = f"ref_{out_c_name}"
            body_lines.append(
                f"    std::vector<{c_type}> {ref_var}({out_c_name}.size());"
            )
            body_lines.append(
                f"    reference_matmul({a_var}, {b_var}, {ref_var}, "
                f"{i_dim}, {k_dim}, {j_dim});"
            )
            value_vars[node.uid] = ref_var
            if node.name in {n.name for n in graph.outputs_nodes}:
                output_ref_vars.append(ref_var)
        else:
            raise NotImplementedError(
                f"runtime_validate export does not support op {op.name!r}"
            )

    if len(output_ref_vars) != len(output_c_names):
        raise NotImplementedError(
            "runtime_validate export could not generate reference for all outputs"
        )

    return helpers, "\n".join(body_lines), output_ref_vars
