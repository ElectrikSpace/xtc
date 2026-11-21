# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A")
b = O.tensor((I,), dtype, name="B")

with O.graph(name="sub_row") as gb:
    O.sub_row(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="sub_row_mlir",
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
# CHECK-NEXT:    func.func @sub_row(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<4xf32> {llvm.noalias}, %arg2: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x32xf32>, memref<4xf32>) outs(%arg2 : memref<4x32xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: f32, %in_0: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.subf %in, %in_0 : f32
# CHECK-NEXT:        linalg.yield %0 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK:       // -----// IR Dump After transform //----- //
# CHECK:       scf.for
# CHECK:       CODE: 0
