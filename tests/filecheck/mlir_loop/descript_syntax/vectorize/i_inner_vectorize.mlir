// RUN: not mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s
func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i",
				"j",
					"k" = {"vectorize"},
						"i#1" = {"vectorize"},
							"k#8",
								"j#32" = {"vectorize"}
		}
	}
	ins(%A, %B: memref<256x512xf64>, memref<512x256xf64>)
	outs(%C: memref<256x256xf64>)
	return
}
// CHECK:  Inner loop on k isn't vectorized but an outer one is.
