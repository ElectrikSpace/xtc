# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import numpy as np

from xtc.backends.mlir import Backend
from xtc.kernelbase import ggml_matmul_q_graph
from xtc.quant import ggml_dequantize, ggml_quantize

M, N, K = 4, 32, 128
graph = ggml_matmul_q_graph(M, N, K, "q4_0", name="matmul_q4_0")
print(graph)

impl = Backend(graph, use_tensor_dialect=True)

sch = impl.get_scheduler()
sch.tile("j", {"j1": 16})
sch.tile("k", {"k1": 32})
sch.interchange(["i", "j", "k", "k1", "j1"])
# Dequantizing pack of B (the weight) at k: the quantized blocks of the tile
# are copied to local buffers, then dequantized to a local 32x16 tile.
sch.pack_at("k", 1, mtype="local")
sch.vectorize(["j1"])
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_q4_0_dequant_pack_mlir_tensor",
    print_source_ir=True,
    print_transformed_ir=True,
    print_bufferization_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

rng = np.random.default_rng(0)
a = rng.standard_normal((M, K)).astype(np.float32)
bq = ggml_quantize(rng.standard_normal((N, K)).astype(np.float32), "q4_0")
c = np.zeros((M, N), np.float32)
module.get_evaluator(parameters=([a, bq], [c])).evaluate()
ref = a @ ggml_dequantize(bq, "q4_0").T
print(f"MATCH: {np.allclose(c, ref, rtol=1e-5, atol=1e-4)}")

# The dequantize op reads the q4_0 blocks through three views of the raw
# buffer: scale low byte, scale high byte, packed nibbles.
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d1, d0 floordiv 32, 0)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d1, d0 floordiv 32, d0 mod 16)>
# CHECK-NEXT:  #map2 = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_q4_0(%arg0: tensor<4x128xf32> {llvm.noalias}, %arg1: tensor<32x4x18xi8> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %0 = tensor.empty() : tensor<128x32xf32>
# CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg1[0, 0, 0] [32, 4, 1] [1, 1, 1] : tensor<32x4x18xi8> to tensor<32x4x1xi8>
# CHECK-NEXT:      %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, 1] [32, 4, 1] [1, 1, 1] : tensor<32x4x18xi8> to tensor<32x4x1xi8>
# CHECK-NEXT:      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 2] [32, 4, 16] [1, 1, 1] : tensor<32x4x18xi8> to tensor<32x4x16xi8>
# CHECK-NEXT:      %1 = linalg.generic {indexing_maps = [#map, #map, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<32x4x1xi8>, tensor<32x4x1xi8>, tensor<32x4x16xi8>) outs(%0 : tensor<128x32xf32>) attrs =  {__xtc_id_B_} {
# CHECK-NEXT:      ^bb0(%in: i8, %in_2: i8, %in_3: i8, %out: f32):
# CHECK:           } -> tensor<128x32xf32>
# CHECK:           %4 = linalg.matmul {__xtc_id_C_} ins(%arg0, %1 : tensor<4x128xf32>, tensor<128x32xf32>) outs(%3 : tensor<4x32xf32>) -> tensor<4x32xf32>

# The pack at k fuses the dequantize op in the k loop, stages its quantized
# operands in local buffers and makes its output a tile-sized tensor.
# CHECK:         %fused_op_6, %new_containing_op_7 = transform.structured.fuse_into_containing_op %5 into %loops_5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:    %6 = transform.get_operand %fused_op_6[0, 1, 2] : (!transform.any_op) -> !transform.any_value
# CHECK-NEXT:    %7 = transform.structured.promote_tensor %6 : !transform.any_value
# CHECK-NEXT:    %8 = transform.get_parent_op %fused_op_6 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:    transform.apply_patterns to %8 {
# CHECK-NEXT:      transform.apply_patterns.tensor.fold_tensor_empty
# CHECK-NEXT:    } : !transform.any_op

# Local buffers are allocated once, then for each tile the quantized blocks
# are copied, dequantized to the local tile, which is read by the matmul.
# CHECK:       // -----// IR Dump After Tensor Lowering //----- //
# CHECK:           func.func @matmul_q4_0(%arg0: memref<4x128xf32> {llvm.noalias}, %arg1: memref<32x4x18xi8> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NOT:       memref.alloc()
# CHECK:           %alloca = memref.alloca() {alignment = 256 : i64} : memref<16x1x1xi8>
# CHECK-NEXT:      %alloca_4 = memref.alloca() {alignment = 256 : i64} : memref<16x1x1xi8>
# CHECK-NEXT:      %alloca_5 = memref.alloca() {alignment = 256 : i64} : memref<16x1x16xi8>
# CHECK-NEXT:      %alloca_6 = memref.alloca() {alignment = 256 : i64} : memref<32x16xf32>
# CHECK:               %3 = scf.for %arg7 = %c0 to %c128 step %c32 iter_args(%arg8 = %subview_13) -> (memref<1x16xf32, strided<[32, 1], offset: ?>>) {
# CHECK:                 memref.copy %subview_16, %alloca : memref<16x1x1xi8, strided<[72, 18, 1], offset: ?>> to memref<16x1x1xi8>
# CHECK:                 memref.copy %subview_17, %alloca_4 : memref<16x1x1xi8, strided<[72, 18, 1], offset: ?>> to memref<16x1x1xi8>
# CHECK:                 memref.copy %subview_18, %alloca_5 : memref<16x1x16xi8, strided<[72, 18, 1], offset: ?>> to memref<16x1x16xi8>
# CHECK-NEXT:            linalg.generic {indexing_maps = [#map2, #map2, #map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%alloca, %alloca_4, %alloca_5 : memref<16x1x1xi8>, memref<16x1x1xi8>, memref<16x1x16xi8>) outs(%alloca_6 : memref<32x16xf32>) attrs =  {__xtc_id_B_} {
# CHECK:                 %6 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %arg8) -> (memref<1x16xf32, strided<[32, 1], offset: ?>>) {
# CHECK:                   %subview_20 = memref.subview %alloca_6[%arg9, 0] [1, 16] [1, 1] : memref<32x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
# CHECK:                   %14 = vector.fma %12, %10, %13 : vector<16xf32>
# CHECK:               } {"./k"}
# CHECK-NOT:       memref.alloc()

# CHECK:       - %2: dequantize(%1, transpose=True, format=q4_0) {name = 'B'} : [32x4x18xuint8] -> [128x32xfloat32]
# CHECK-NEXT:  - %3: matmul(%0, %2) {name = 'C'} : [4x128xfloat32, 128x32xfloat32] -> [4x32xfloat32]
# CHECK:       CODE: 0
# CHECK-NEXT:  MATCH: True
