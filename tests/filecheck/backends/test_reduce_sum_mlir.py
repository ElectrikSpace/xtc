# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A")

with O.graph(name="reduce_sum") as gb:
    O.reduce_sum(a, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="reduce_sum_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  #map1 = affine_map<(d0, d1) -> (d0)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @reduce_sum(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<4xf32> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%cst : f32) outs(%arg1 : memref<4xf32>)
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<4x32xf32>) outs(%arg1 : memref<4xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.addf %in, %out : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK:       // -----// IR Dump After transform //----- //
# CHECK:       scf.for
# CHECK:       CODE: 0
