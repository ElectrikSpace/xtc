// RUN: mlir-loop --no-alias --print-transformed-ir %s 2>&1 | grep "vector\.fma" | filecheck %s

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  linalg.matmul
    {
      loop.dims = ["I","J","K"],
      loop.schedule = {
        "I" = {"Parallel"},
          "J",
            "K",
              "I#1" = {"Unroll"},
                "K#8"= {"Unroll"},
                  "J#64" = {"Vectorize"}
      }
    }
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
// CHECK:                 %7 = vector.fma %5, %3, %6 : vector<64xf32>
// CHECK-NEXT:            %16 = vector.fma %14, %12, %15 : vector<64xf32>
// CHECK-NEXT:            %25 = vector.fma %23, %21, %24 : vector<64xf32>
// CHECK-NEXT:            %34 = vector.fma %32, %30, %33 : vector<64xf32>
// CHECK-NEXT:            %43 = vector.fma %41, %39, %42 : vector<64xf32>
// CHECK-NEXT:            %52 = vector.fma %50, %48, %51 : vector<64xf32>
// CHECK-NEXT:            %61 = vector.fma %59, %57, %60 : vector<64xf32>
// CHECK-NEXT:            %70 = vector.fma %68, %66, %69 : vector<64xf32>
