# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul_relu") as gb:
    m = O.matmul(a, b, name="matmul")
    O.relu(m, name="relu")

graph = gb.graph
print(graph)

impl = Backend(graph, always_vectorize=False, no_alias=True)

sch = impl.get_scheduler(nodes=["matmul"])
sch.tile("i", {"i1": 2})
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_relu_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map1 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_matmul_} ins(%arg0, %arg1 : memref<4x512xf32>, memref<512x32xf32>) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %alloca [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:      %collapse_shape_0 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%collapse_shape, %cst_1 : memref<128xf32>, f32) outs(%collapse_shape_0 : memref<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_matmul_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "i1" : !transform.any_op
# CHECK-NEXT:      transform.structured.vectorize %tiled_linalg_op_4 : !transform.any_op
# CHECK-NEXT:      transform.loop.unroll %loops_5 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, 0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (0, d1, d0)>
# CHECK-NEXT:  #map2 = affine_map<(d0) -> (d0)>
# CHECK-NEXT:  #map3 = affine_map<(d0) -> ()>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul_relu(%arg0: memref<4x512xf32> {llvm.noalias}, %arg1: memref<512x32xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      %alloca = memref.alloca() {alignment = 256 : i64} : memref<4x32xf32>
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_matmul_0_} ins(%cst : f32) outs(%alloca : memref<4x32xf32>)
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xf32> to memref<4x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_2 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_3 = memref.subview %alloca[0, 0] [4, 32] [1, 1] : memref<4x32xf32> to memref<4x32xf32, strided<[32, 1]>>
# CHECK-NEXT:        %c0_4 = arith.constant 0 : index
# CHECK-NEXT:        %c4 = arith.constant 4 : index
# CHECK-NEXT:        %c2 = arith.constant 2 : index
# CHECK-NEXT:        scf.for %arg4 = %c0_4 to %c4 step %c2 {
# CHECK-NEXT:          %subview_5 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_6 = memref.subview %subview_2[0, 0] [1, 32] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %subview_7 = memref.subview %subview_3[%arg4, 0] [2, 32] [1, 1] : memref<4x32xf32, strided<[32, 1]>> to memref<2x32xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          %c0_8 = arith.constant 0 : index
# CHECK-NEXT:          %c32 = arith.constant 32 : index
# CHECK-NEXT:          %c16 = arith.constant 16 : index
# CHECK-NEXT:          scf.for %arg5 = %c0_8 to %c32 step %c16 {
# CHECK-NEXT:            %subview_9 = memref.subview %subview_5[0, 0] [2, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<2x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_10 = memref.subview %subview_6[0, %arg5] [1, 16] [1, 1] : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_11 = memref.subview %subview_7[0, %arg5] [2, 16] [1, 1] : memref<2x32xf32, strided<[32, 1], offset: ?>> to memref<2x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c0_12 = arith.constant 0 : index
# CHECK-NEXT:            %c2_13 = arith.constant 2 : index
# CHECK-NEXT:            %c1_14 = arith.constant 1 : index
# CHECK-NEXT:            %c2_15 = arith.constant 2 : index
# CHECK-NEXT:            %subview_16 = memref.subview %subview_9[%c0_12, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_17 = memref.subview %subview_10[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_18 = memref.subview %subview_11[%c0_12, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_19 = arith.constant 1 : index
# CHECK-NEXT:            %c16_20 = arith.constant 16 : index
# CHECK-NEXT:            %c1_21 = arith.constant 1 : index
# CHECK-NEXT:            %c0_22 = arith.constant 0 : index
# CHECK-NEXT:            %cst_23 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %0 = vector.transfer_read %subview_16[%c0_22, %c0_22], %cst_23 {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_24 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %1 = vector.transfer_read %subview_17[%c0_22, %c0_22], %cst_24 {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_25 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %2 = vector.transfer_read %subview_18[%c0_22, %c0_22], %cst_25 : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %3 = arith.mulf %0, %1 : vector<1x16x1xf32>
# CHECK-NEXT:            %4 = vector.multi_reduction <add>, %3, %2 [2] : vector<1x16x1xf32> to vector<1x16xf32>
# CHECK-NEXT:            %c0_26 = arith.constant 0 : index
# CHECK-NEXT:            vector.transfer_write %4, %subview_18[%c0_26, %c0_26] : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_27 = arith.constant 1 : index
# CHECK-NEXT:            %5 = arith.muli %c1_14, %c1_27 : index
# CHECK-NEXT:            %6 = arith.addi %c0_12, %5 : index
# CHECK-NEXT:            %subview_28 = memref.subview %subview_9[%6, 0] [1, 1] [1, 1] : memref<2x1xf32, strided<[512, 1], offset: ?>> to memref<1x1xf32, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_29 = memref.subview %subview_10[0, 0] [1, 16] [1, 1] : memref<1x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_30 = memref.subview %subview_11[%6, 0] [1, 16] [1, 1] : memref<2x16xf32, strided<[32, 1], offset: ?>> to memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %c1_31 = arith.constant 1 : index
# CHECK-NEXT:            %c16_32 = arith.constant 16 : index
# CHECK-NEXT:            %c1_33 = arith.constant 1 : index
# CHECK-NEXT:            %c0_34 = arith.constant 0 : index
# CHECK-NEXT:            %cst_35 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %7 = vector.transfer_read %subview_28[%c0_34, %c0_34], %cst_35 {in_bounds = [false, true, false], permutation_map = #map} : memref<1x1xf32, strided<[512, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_36 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %8 = vector.transfer_read %subview_29[%c0_34, %c0_34], %cst_36 {in_bounds = [true, false, false], permutation_map = #map1} : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16x1xf32>
# CHECK-NEXT:            %cst_37 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:            %9 = vector.transfer_read %subview_30[%c0_34, %c0_34], %cst_37 : memref<1x16xf32, strided<[32, 1], offset: ?>>, vector<1x16xf32>
# CHECK-NEXT:            %10 = arith.mulf %7, %8 : vector<1x16x1xf32>
# CHECK-NEXT:            %11 = vector.multi_reduction <add>, %10, %9 [2] : vector<1x16x1xf32> to vector<1x16xf32>
# CHECK-NEXT:            %c0_38 = arith.constant 0 : index
# CHECK-NEXT:            vector.transfer_write %11, %subview_30[%c0_38, %c0_38] : vector<1x16xf32>, memref<1x16xf32, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {j}
# CHECK-NEXT:        } {i}
# CHECK-NEXT:      } {k}
# CHECK-NEXT:      %collapse_shape = memref.collapse_shape %alloca [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:      %collapse_shape_0 = memref.collapse_shape %arg2 [[0, 1]] : memref<4x32xf32> into memref<128xf32>
# CHECK-NEXT:      %cst_1 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel"]} ins(%collapse_shape, %cst_1 : memref<128xf32>, f32) outs(%collapse_shape_0 : memref<128xf32>) attrs =  {__xtc_id_relu_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_2: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.maximumf %in, %in_2 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul_relu
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xfloat32
# CHECK-NEXT:    - %1 : 512x32xfloat32
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %3 : 4x32xfloat32
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'matmul'} : [4x512xfloat32, 512x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:    - %3: relu(%2) {name = 'relu'} : [4x32xfloat32] -> [4x32xfloat32]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
