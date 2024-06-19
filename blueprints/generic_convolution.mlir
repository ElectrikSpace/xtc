#map = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
module {
  func.func @payload0(%arg0: memref<2x4x5x2xf32>, %arg1: memref<2x2x2x3xf32>) -> memref<2x3x4x2x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3x4x2x3xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%cst : f32) outs(%alloc : memref<2x3x4x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    
    // Conv2D
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d3, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
      ],
      iterator_types = [
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "reduction",
        "reduction"
      ]} ins(%arg0, %arg1 : memref<2x4x5x2xf32>, memref<2x2x2x3xf32>) outs(%alloc : memref<2x3x4x2x3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.mulf %in, %in_0 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    }
    
    return %alloc : memref<2x3x4x2x3xf32>
  }
}
