#! /usr/bin/python3

import sys
import subprocess
import re

INDENT = 7

def process_file(file_path: str, in_place: bool, regexp: bool):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines[0].startswith('// RUN:'):
        print(f"Error: The first line of the file does not start with '// RUN:'")
        return

    # Extract the command from the first line
    first_line = lines[0].strip()
    command = re.sub(r'// RUN:\s*', '', first_line)
    command = command.split('|')[0].strip()
    command = command.replace('%s', file_path)

    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Remove trailing newline if present
    if output.endswith('\n'):
        output = output[:-1]

    # Process the output
    output_lines = output.splitlines()
    processed_output = '// CHECK:' + INDENT*' ' + output_lines[0] + '\n'
    for line in output_lines[1:]:
        processed_output += '// CHECK-NEXT:' + (INDENT - 5)*' ' + line + '\n'

    if in_place:
        # Remove lines starting with '// CHECK:' or '// CHECK-NEXT:'
        new_lines = [line for line in lines if not line.startswith('// CHECK:') and not line.startswith('// CHECK-NEXT:')]
        new_lines.append(processed_output)

        with open(file_path, 'w') as file:
            file.writelines(new_lines)
    else:
        print(processed_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Filecheck directives.",
        epilog = "Apply to all MLIR files in <dir>: for file in <dir>/*.mlir; do python3 gen_filecheck.py \"$file\" -i; done"
    )
    parser.add_argument('src', type=str, help='Source file.')
    parser.add_argument('-i', action='store_true', help='Insert the resulting Filecheck directives.')
    parser.add_argument('-r', action='store_true', help='Replace MLIR/LLVM variables by regexps.')
    args = parser.parse_args()

    process_file(args.src, args.i, args.r)
