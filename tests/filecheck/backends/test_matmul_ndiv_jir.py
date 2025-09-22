# RUN: python %s 2>&1 | filecheck %s
# REQUIRES: module_jir

import xtc.graphs.xtc.op as O
from xtc.backends.jir import JIRBackend as Backend

I, J, K, dtype = 4, 32, 512, "float32"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.tile("i", {"i1": 3}) # non-divisible tile
sch.tile("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})    # non-full/non-divisible unrolled
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    dump_file="matmul_ndiv_jir",
    print_source_ir=True,
    print_transformed_ir=True,
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")
# XFAIL: *
