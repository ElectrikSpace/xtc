import argparse
import os
import re
import csv
import json

INPUT_R = re.compile(r"(.*)-(.*)\.yml\.csv")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Path to the folder containing results")
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     help="Path to the output folder"
    # )
    parser.add_argument("-c", type=int, default=1, help="number of cores")

    args = parser.parse_args()

    input = args.input
    cores = args.c

    output = input + "/results_json/"

    if not os.path.isdir(output):
        os.mkdir(output)

    for f in os.listdir(input):
        if f.endswith(".csv"):
            m = INPUT_R.match(f)
            if not m:
                print(f"WARNING: file {f} does not match regex.")
                continue
            op, sched = m.groups()
            op = op.lower()
            out_f = f"{output}/results.c{cores}.{op}.{sched}.2048.1.jsonl"
            with open(f"{input}/{f}", "r") as f_in, open(out_f, "w") as f_out:
                data_in = csv.DictReader(f_in)
                data_out = [{"results": [float(r["time"])]} for r in data_in]
                # print(data_out)
                for result in data_out:
                    json.dump(result, f_out)

            with open(out_f, "r+") as f_out:
                data_in = f_out.read()
                data_out = data_in.replace("}", "}\n")
                f_out.seek(0)
                f_out.write(data_out)
                f_out.truncate()


if __name__ == "__main__":
    main()
