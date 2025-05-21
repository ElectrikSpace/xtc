// RUN: mlir-loop --old-syntax %s --vectors-size 8 --no-alias --print-transformed-ir 2>&1 | filecheck %s

func.func @myfun(
  %I: memref<1x30x30x64xf32>,
  %K: memref<3x3x64x128xf32>,
  %O: memref<1x28x28x128xf32>
) {
  linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
      ],
      iterator_types = [
        "parallel",
        "parallel",
        "parallel",
        "parallel",
        "reduction",
        "reduction",
        "reduction"
      ]
  }
  ins (%I, %K : memref<1x30x30x64xf32>, memref<3x3x64x128xf32>)
  outs(%O : memref<1x28x28x128xf32>)
  attrs = {
    loop.dims = ["n","h","w","f","r","s","c"],
    loop.vectorize = ["n"]
  }
  {
    ^bb0(%0: f32, %1: f32, %2: f32) :
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      linalg.yield %4 : f32
  }
  return
}

// CHECK:       // -----// IR Dump After transform //----- //
// CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-NEXT:  #map1 = affine_map<(d0, d1, d2, d3) -> (0)>
// CHECK-NEXT:  module attributes {transform.with_named_sequence} {
// CHECK-NEXT:    func.func @myfun(%arg0: memref<1x30x30x64xf32> {llvm.noalias}, %arg1: memref<3x3x64x128xf32> {llvm.noalias}, %arg2: memref<1x28x28x128xf32> {llvm.noalias}) {
// CHECK-NEXT:      affine.for %arg3 = 0 to 1 {
// CHECK-NEXT:        affine.for %arg4 = 0 to 28 {
// CHECK-NEXT:          affine.for %arg5 = 0 to 28 {
// CHECK-NEXT:            affine.for %arg6 = 0 to 128 step 8 {
// CHECK-NEXT:              affine.for %arg7 = 0 to 3 {
// CHECK-NEXT:                affine.for %arg8 = 0 to 3 {
// CHECK-NEXT:                  affine.for %arg9 = 0 to 64 {
// CHECK-NEXT:                    %0 = affine.apply #map(%arg4, %arg7)
// CHECK-NEXT:                    %1 = affine.apply #map(%arg5, %arg8)
// CHECK-NEXT:                    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:                    %2 = vector.transfer_read %arg0[%arg3, %0, %1, %arg9], %cst {in_bounds = [true], permutation_map = #map1} : memref<1x30x30x64xf32>, vector<8xf32>
// CHECK-NEXT:                    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:                    %3 = vector.transfer_read %arg1[%arg7, %arg8, %arg9, %arg6], %cst_0 : memref<3x3x64x128xf32>, vector<8xf32>
// CHECK-NEXT:                    %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:                    %4 = vector.transfer_read %arg2[%arg3, %arg4, %arg5, %arg6], %cst_1 : memref<1x28x28x128xf32>, vector<8xf32>
// CHECK-NEXT:                    %5 = arith.mulf %2, %3 : vector<8xf32>
// CHECK-NEXT:                    %6 = arith.addf %4, %5 : vector<8xf32>
// CHECK-NEXT:                    vector.transfer_write %6, %arg2[%arg3, %arg4, %arg5, %arg6] : vector<8xf32>, memref<1x28x28x128xf32>
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
