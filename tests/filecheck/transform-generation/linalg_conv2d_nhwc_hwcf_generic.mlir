// RUN: mlir-loop %s --no-alias --print-source-ir 2>&1 | filecheck %s

func.func @myfun(
  %I: memref<1x30x30x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x28x128xf32>
) {
  %cst = arith.constant 0.000 : f32
  linalg.fill
    {
      loop.dims = ["n","h","w","f"],
      loop.tiles = {"f" = {"f1" = 8}},
      loop.interchange = ["n","h","w","f","f1"],
      loop.vectorize = ["f1"],
      loop.parallelize = ["h"]
    }
    ins(%cst : f32)
    outs(%O : memref<1x28x28x128xf32>)
  linalg.generic {
      indexing_maps = [
        affine_map<(n,h,w,f,r,s,c) -> (n,h+r,w+s,c)>,
        affine_map<(n,h,w,f,r,s,c) -> (r,s,c,f)>,
        affine_map<(n,h,w,f,r,s,c) -> (n,h,w,f)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
          "reduction", "reduction", "reduction"],
      loop.dims = ["n","h","w","f","c","r","s"],
      loop.tiles = {
        "f" = {"f1" = 64},
        "h" = {"h1" = 14, "h2" = 2},
        "w" = {"w1" = 2},
        "c" = {"c1" = 4}
      },
      loop.interchange = ["r", "s", "h", "w", "h1", "f", "c", "c1", "w1", "h2", "f1"],
      loop.vectorize = ["f1"],
      loop.unroll = {f1 = 4, h2 = 2, w1 = 2, c1 = 4}
    }
    ins(%I, %K : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>)
    outs(%O : memref<1x28x28x128xf32>)
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}
// CHECK:       // -----// IR Dump Before transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
// CHECK-NEXT:  #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      linalg.fill {__id0__} ins(%cst : f32) outs(%arg2 : memref<1x28x28x128xf32>)
// CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>) outs(%arg2 : memref<1x28x28x128xf32>) {
// CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
// CHECK-NEXT:        %0 = arith.mulf %in, %in_0 : f32
// CHECK-NEXT:        %1 = arith.addf %out, %0 : f32
// CHECK-NEXT:        linalg.yield %1 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
// CHECK-NEXT:      %0 = transform.structured.match attributes {__id0__} in %arg0 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops "__id0__n" : !transform.any_op
// CHECK-NEXT:      %tiled_op, %forall_op = transform.structured.tile_using_forall %tiled_linalg_op tile_sizes [0, 1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %forall_op "__id0__h" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_op tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_1 "__id0__w" : !transform.any_op
// CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [0, 0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
// CHECK-NEXT:      transform.annotate %loops_3 "__id0__f" : !transform.any_op
// CHECK-NEXT:      %1 = transform.get_parent_op %loops {isolated_from_above} : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:      transform.apply_patterns to %2 {
// CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
// CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
// CHECK-NEXT:      } : !transform.any_op
// CHECK-NEXT:      transform.yield 
// CHECK-NEXT:    }
// CHECK-NEXT:  }
