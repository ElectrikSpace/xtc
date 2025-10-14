// RUN: not mlir-loop --no-alias --print-source-ir %s 2>&1 | filecheck %s

func.func @matmul(%A: memref<256x512xf64>, %B: memref<512x256xf64>, %C: memref<256x256xf64>){
	linalg.matmul {
		loop.dims = ["i", "j", "k"],
		loop.schedule = {
			"i[0:5]" = { "j", "k" },
				"i[5:]" = { "j", "k", "l" }
		}
	}
	ins(%A, %B : memref<256x512xf64>, memref<512x256xf64>)
	outs(%C: memref<256x256xf64>)
	return
}
// CHECK: Axis l is not a defined axis. 
// CHECK: Known axis are: ['i', 'j', 'k']
