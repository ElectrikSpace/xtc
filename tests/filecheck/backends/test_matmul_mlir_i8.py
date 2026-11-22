# RUN: python %s 2>&1 | filecheck %s

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend

I, J, K, dtype = 4, 32, 512, "int8"
a = O.tensor((I, K), dtype, name="A")
b = O.tensor((K, J), dtype, name="B")

with O.graph(name="matmul") as gb:
    O.matmul(a, b, name="C")

graph = gb.graph
print(graph)

impl = Backend(graph)

sch = impl.get_scheduler()
sch.strip_mine("i", {"i1": 2})
sch.strip_mine("j", {"j1": 16})
sch.interchange(["k", "i", "j", "i1", "j1"])
sch.vectorize(["j1"])
sch.unroll({"i1": 2})
sched = sch.schedule()

comp = impl.get_compiler(
    shared_lib=True,
    print_source_ir=True,
    print_transformed_ir=True,
    dump_file="matmul_i8_mlir",
)
module = comp.compile(sched)
executor = module.get_executor(validate=True)
res = executor.execute()
print(f"CODE: {res}")

# CHECK:       // -----// IR Dump Before transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xi8> {llvm.noalias}, %arg1: memref<512x32xi8> {llvm.noalias}, %arg2: memref<4x32xi8> {llvm.noalias}) {
# CHECK-NEXT:      %c0_i8 = arith.constant 0 : i8
# CHECK-NEXT:      linalg.fill {__xtc_id_C_0_} ins(%c0_i8 : i8) outs(%arg2 : memref<4x32xi8>)
# CHECK-NEXT:      linalg.matmul {__xtc_id_C_} ins(%arg0, %arg1 : memref<4x512xi8>, memref<512x32xi8>) outs(%arg2 : memref<4x32xi8>)
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @_vecto(%arg0: !transform.any_op {transform.consumed}) {
# CHECK-NEXT:      transform.structured.vectorize %arg0 : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
# CHECK-NEXT:      %0 = transform.structured.match attributes {__xtc_id_C_0_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_1 "./j" : !transform.any_op
# CHECK-NEXT:      %1 = transform.structured.match attributes {__xtc_id_C_} in %arg0 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_2, %loops_3 = transform.structured.tile_using_for %1 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_3 "./k" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_4, %loops_5 = transform.structured.tile_using_for %tiled_linalg_op_2 tile_sizes [2, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_5 "./i" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_6, %loops_7 = transform.structured.tile_using_for %tiled_linalg_op_4 tile_sizes [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_7 "./j" : !transform.any_op
# CHECK-NEXT:      %tiled_linalg_op_8, %loops_9 = transform.structured.tile_using_for %tiled_linalg_op_6 tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
# CHECK-NEXT:      transform.annotate %loops_9 "./i1" : !transform.any_op
# CHECK-NEXT:      %2 = transform.get_parent_op %tiled_linalg_op_8 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %2 {
# CHECK-NEXT:        transform.apply_patterns.xtc.fold_unit_extent_dims_via_slices_for_vectorization
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      %3 = transform.structured.match interface{LinalgOp} in %2 : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.include @_vecto failures(suppress) (%3) : (!transform.any_op) -> ()
# CHECK-NEXT:      transform.loop.unroll %loops_9 {factor = 2 : i64} : !transform.any_op
# CHECK-NEXT:      %4 = transform.get_parent_op %loops_3 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.reduction_to_contract
# CHECK-NEXT:        transform.apply_patterns.vector.transfer_permutation_patterns
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.apply_patterns to %4 {
# CHECK-NEXT:        transform.apply_patterns.vector.lower_outerproduct
# CHECK-NEXT:        transform.apply_patterns.vector.lower_contraction
# CHECK-NEXT:      } : !transform.any_op
# CHECK-NEXT:      transform.yield 
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  // -----// IR Dump After transform //----- //
# CHECK-NEXT:  module attributes {transform.with_named_sequence} {
# CHECK-NEXT:    func.func @matmul(%arg0: memref<4x512xi8> {llvm.noalias}, %arg1: memref<512x32xi8> {llvm.noalias}, %arg2: memref<4x32xi8> {llvm.noalias}) {
# CHECK-NEXT:      %cst = arith.constant dense<0> : vector<1x16xi8>
# CHECK-NEXT:      %0 = ub.poison : i8
# CHECK-NEXT:      %c16 = arith.constant 16 : index
# CHECK-NEXT:      %c2 = arith.constant 2 : index
# CHECK-NEXT:      %c512 = arith.constant 512 : index
# CHECK-NEXT:      %c32 = arith.constant 32 : index
# CHECK-NEXT:      %c0_i8 = arith.constant 0 : i8
# CHECK-NEXT:      %c0 = arith.constant 0 : index
# CHECK-NEXT:      %c4 = arith.constant 4 : index
# CHECK-NEXT:      %c1 = arith.constant 1 : index
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c4 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg2[%arg3, 0] [1, 32] [1, 1] : memref<4x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c32 step %c1 {
# CHECK-NEXT:          %subview_0 = memref.subview %subview[0, %arg4] [1, 1] [1, 1] : memref<1x32xi8, strided<[32, 1], offset: ?>> to memref<1x1xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          linalg.fill {__xtc_id_C_0_} ins(%c0_i8 : i8) outs(%subview_0 : memref<1x1xi8, strided<[32, 1], offset: ?>>)
# CHECK-NEXT:        } {"./j"}
# CHECK-NEXT:      } {"./i"}
# CHECK-NEXT:      scf.for %arg3 = %c0 to %c512 step %c1 {
# CHECK-NEXT:        %subview = memref.subview %arg0[0, %arg3] [4, 1] [1, 1] : memref<4x512xi8> to memref<4x1xi8, strided<[512, 1], offset: ?>>
# CHECK-NEXT:        %subview_0 = memref.subview %arg1[%arg3, 0] [1, 32] [1, 1] : memref<512x32xi8> to memref<1x32xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:        %subview_1 = memref.subview %arg2[0, 0] [4, 32] [1, 1] : memref<4x32xi8> to memref<4x32xi8, strided<[32, 1]>>
# CHECK-NEXT:        scf.for %arg4 = %c0 to %c4 step %c2 {
# CHECK-NEXT:          %subview_2 = memref.subview %subview[%arg4, 0] [2, 1] [1, 1] : memref<4x1xi8, strided<[512, 1], offset: ?>> to memref<2x1xi8, strided<[512, 1], offset: ?>>
# CHECK-NEXT:          %subview_3 = memref.subview %subview_1[%arg4, 0] [2, 32] [1, 1] : memref<4x32xi8, strided<[32, 1]>> to memref<2x32xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          scf.for %arg5 = %c0 to %c32 step %c16 {
# CHECK-NEXT:            %subview_4 = memref.subview %subview_0[0, %arg5] [1, 16] [1, 1] : memref<1x32xi8, strided<[32, 1], offset: ?>> to memref<1x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_5 = memref.subview %subview_3[0, %arg5] [2, 16] [1, 1] : memref<2x32xi8, strided<[32, 1], offset: ?>> to memref<2x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_6 = memref.subview %subview_2[%c0, 0] [1, 1] [1, 1] : memref<2x1xi8, strided<[512, 1], offset: ?>> to memref<1x1xi8, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_7 = memref.subview %subview_5[%c0, 0] [1, 16] [1, 1] : memref<2x16xi8, strided<[32, 1], offset: ?>> to memref<1x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %1 = vector.transfer_read %subview_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xi8, strided<[512, 1], offset: ?>>, vector<1x1xi8>
# CHECK-NEXT:            %2 = vector.transfer_read %subview_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xi8, strided<[32, 1], offset: ?>>, vector<1x16xi8>
# CHECK-NEXT:            %3 = vector.transfer_read %subview_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xi8, strided<[32, 1], offset: ?>>, vector<1x16xi8>
# CHECK-NEXT:            %4 = vector.extract %2[0] : vector<16xi8> from vector<1x16xi8>
# CHECK-NEXT:            %5 = vector.extract %1[0, 0] : i8 from vector<1x1xi8>
# CHECK-NEXT:            %6 = vector.broadcast %5 : i8 to vector<16xi8>
# CHECK-NEXT:            %7 = vector.extract %3[0] : vector<16xi8> from vector<1x16xi8>
# CHECK-NEXT:            %8 = arith.muli %6, %4 : vector<16xi8>
# CHECK-NEXT:            %9 = arith.addi %8, %7 : vector<16xi8>
# CHECK-NEXT:            %10 = vector.insert %9, %cst [0] : vector<16xi8> into vector<1x16xi8>
# CHECK-NEXT:            vector.transfer_write %10, %subview_7[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xi8>, memref<1x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %subview_8 = memref.subview %subview_2[%c1, 0] [1, 1] [1, 1] : memref<2x1xi8, strided<[512, 1], offset: ?>> to memref<1x1xi8, strided<[512, 1], offset: ?>>
# CHECK-NEXT:            %subview_9 = memref.subview %subview_5[%c1, 0] [1, 16] [1, 1] : memref<2x16xi8, strided<[32, 1], offset: ?>> to memref<1x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:            %11 = vector.transfer_read %subview_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x1xi8, strided<[512, 1], offset: ?>>, vector<1x1xi8>
# CHECK-NEXT:            %12 = vector.transfer_read %subview_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xi8, strided<[32, 1], offset: ?>>, vector<1x16xi8>
# CHECK-NEXT:            %13 = vector.transfer_read %subview_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<1x16xi8, strided<[32, 1], offset: ?>>, vector<1x16xi8>
# CHECK-NEXT:            %14 = vector.extract %12[0] : vector<16xi8> from vector<1x16xi8>
# CHECK-NEXT:            %15 = vector.extract %11[0, 0] : i8 from vector<1x1xi8>
# CHECK-NEXT:            %16 = vector.broadcast %15 : i8 to vector<16xi8>
# CHECK-NEXT:            %17 = vector.extract %13[0] : vector<16xi8> from vector<1x16xi8>
# CHECK-NEXT:            %18 = arith.muli %16, %14 : vector<16xi8>
# CHECK-NEXT:            %19 = arith.addi %18, %17 : vector<16xi8>
# CHECK-NEXT:            %20 = vector.insert %19, %cst [0] : vector<16xi8> into vector<1x16xi8>
# CHECK-NEXT:            vector.transfer_write %20, %subview_9[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xi8>, memref<1x16xi8, strided<[32, 1], offset: ?>>
# CHECK-NEXT:          } {"./j"}
# CHECK-NEXT:        } {"./i"}
# CHECK-NEXT:      } {"./k"}
# CHECK-NEXT:      return
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  
# CHECK-NEXT:  graph:
# CHECK-NEXT:    name: matmul
# CHECK-NEXT:    inputs:
# CHECK-NEXT:    - %0 : 4x512xint8
# CHECK-NEXT:    - %1 : 512x32xint8
# CHECK-NEXT:    outputs:
# CHECK-NEXT:    - %2 : 4x32xint8
# CHECK-NEXT:    nodes:
# CHECK-NEXT:    - %2: matmul(%0, %1) {name = 'C'} : [4x512xint8, 512x32xint8] -> [4x32xint8]
# CHECK-NEXT:  
# CHECK-NEXT:  CODE: 0
