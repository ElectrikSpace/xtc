# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_mlir_mppa
# REQUIRES: mlir-target=mppa

import xtc.graphs.xtc.op as O
from xtc.backends.mlir.MlirGraphBackend import MlirGraphBackend as Backend

from xtc.runtimes.accelerator.mppa import MppaDevice

I, J, K, dtype = 4, 8, 16, "float32"
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
sch.tile("k", {"k1": 4})
sch.pack_at("k1", 1, pad=True)
sched = sch.schedule()

mppa = MppaDevice()

comp = impl.get_compiler(
    target=mppa,
    shared_lib=False,
    print_lowered_ir=True,
    dump_file="matmul_mlir_mppa_pack_pad",
)
try:
    comp.compile(sched)
except NotImplementedError as exc:
    print(f"PACK_PAD_NOT_IMPL: {exc}")

# CHECK:       graph:
# CHECK-NEXT:    name: matmul
# CHECK:       PACK_PAD_NOT_IMPL: {{.*}}local_buffer_at{{.*}}
