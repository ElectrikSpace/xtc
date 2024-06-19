from setup_mlir_conv import generic_conv0

impl = generic_conv0()

e = impl.evaluate(
    print_source_ir=False,
    print_transformed_ir=False,
    print_assembly=True,
    color = True,
    debug = False,
)

print(e)
