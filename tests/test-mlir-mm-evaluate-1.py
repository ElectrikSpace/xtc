from setup_mlir_mm import mm1

impl = mm1()

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
