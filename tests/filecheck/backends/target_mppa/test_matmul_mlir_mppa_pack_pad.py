# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_mppa
# REQUIRES: mlir-target=mppa

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend
from xtc.utils.pad import pad_divisible_by, pad_modulo_value

from xtc.runtimes.accelerator.mppa import MppaDevice

I, J, K, dtype = 6, 8, 16, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.define_memory_mesh(axes={"mx": 1, "my": 1})
sch.define_processor_mesh(axes={"px": 1, "py": 1, "psx": 1, "psy": 1})
sch.tile("i", {"i1": 8})
sch.interchange(["k", "j", "i", "i1"])
# A: i is zero filled to 8 (computed on), k is only padded in the layout
sch.pack_at(
    "i",
    0,
    pad=True,
    pad_strategies=[
        [pad_divisible_by(8, zero=True)],
        [pad_modulo_value(32, 20)],
    ],
)
# C: default padding, zero filled to the innermost tile of i
sch.buffer_at("i", pad=True)
sched = sch.schedule()

mppa = MppaDevice()

comp = impl.get_compiler(
    target=mppa,
    shared_lib=False,
    print_transformed_ir=True,
    dump_file="matmul_mlir_mppa_pack_pad",
)
comp.compile(sched)
print("COMPILED")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK:       memref.alloc() {sdist.pad_fill = array<i64: 8, 16>} : memref<8x20xf32, 2>
# CHECK:       memref.subview %{{.*}}[0, 0] [6, 16] [1, 1] : memref<8x20xf32, 2>
# CHECK:       linalg.fill
# CHECK:       memref.alloc() {sdist.pad_fill = array<i64: 8, 8>} : memref<8x8xf32, 2>
# CHECK:       linalg.matmul {{.*}} ins({{.*}} : memref<8x16xf32, strided<[20, 1]>, 2>, {{.*}}) outs({{.*}} : memref<8x8xf32, 2>)
# CHECK:       COMPILED
