module attributes {transform.with_named_sequence} {

func.func @coalesce_inner() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index

  // CHECK: scf.for %[[IV0:.+]]
  // CHECK:   scf.for %[[IV1:.+]]
  // CHECK:     scf.for %[[IV2:.+]]
  // CHECK-NOT:   scf.for %[[IV3:.+]]
  scf.for %i = %c0 to %c10 step %c1 {
    scf.for %j = %c0 to %c10 step %c1 {
      scf.for %k = %i to %j step %c2 {
        // Inner loop must have been removed.
        scf.for %l = %i to %j step %c3 {
        scf.for %m = %i to %j step %c3 {
          arith.addi %i, %j : index
        }
        }
      } {coalesce}
    }
  }
  return
}

  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"scf.for">
    %2 = transform.loop.coalesce %1: (!transform.op<"scf.for">) -> (!transform.op<"scf.for">)
    transform.yield
  }
}
