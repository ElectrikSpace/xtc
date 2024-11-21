// RUN: mlir-loop %s --evaluate --no-alias

func.func @myfun(
  %A: memref<256x512xf32>,
  %B: memref<512x256xf32>,
  %C: memref<256x256xf32>
) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.fill
    {loop.dims = {"i"=256,"j"=256}}
    ins(%cst : f32)
    outs(%C : memref<256x256xf32>)
  linalg.matmul
    {loop.dims = {"i"=256,"j"=256,"k"=512}, loop.unroll = {"k" = 8}}
    ins(%A, %B : memref<256x512xf32>, memref<512x256xf32>)
    outs(%C : memref<256x256xf32>)
  return
}
