# RUN: python %s 2>&1 | filecheck %s
# UNSUPPORTED: mlir-target=nvgpu

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, dtype = 4, 32, "float32"
a = O.tensor((I, J), dtype, name="A")

with O.graph(name="silu") as gb:
    O.silu(a, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="silu_mlir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  #map = affine_map<(d0, d1) -> (d0, d1)>
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @silu(%arg0: memref<4x32xf32> {llvm.noalias}, %arg1: memref<4x32xf32> {llvm.noalias}) {
# CHECK-NEXT:      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<4x32xf32>) outs(%arg1 : memref<4x32xf32>) attrs =  {__xtc_id_C_} {
# CHECK-NEXT:      ^bb0(%in: f32, %out: f32):
# CHECK-NEXT:        %0 = arith.negf %in : f32
# CHECK-NEXT:        %1 = math.exp %0 : f32
# CHECK-NEXT:        %cst = arith.constant 1.000000e+00 : f32
# CHECK-NEXT:        %2 = arith.addf %cst, %1 : f32
# CHECK-NEXT:        %3 = arith.divf %in, %2 : f32
# CHECK-NEXT:        linalg.yield %3 : f32
# CHECK-NEXT:      }
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK:       // -----// IR Dump After transform //----- //
# CHECK:       scf.for
# CHECK:       CODE: 0
