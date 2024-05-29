# Xdsl Transform

## Installation instructions

The previous version relied on XDSL (file Implementer.py). The current
one relies on upstream MLIR bindings.

### Install the right version of MLIR

Install dependencies (Debian):
```
sudo apt install pybind11-dev
```

Choose the commit for which xdsl is made (patches are in the directory
xdsl-transform/patches):
```
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
git checkout 98e674c9f16d677d95c67bc130e267fae331e43c
git apply /path/to/each/patch
```

Compile MLIR/CLANG and the MLIR python bindings:
```
pip install -r mlir/python/requirements.txt
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/bin/llvm-xdsl -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j4
make install
```

Add the Python bindings to your PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:$HOME/bin/llvm-xdsl/python_packages/mlir_core
```

## Install requirements

```
pip install -r requirements.txt
```

For using tvm backend, install TVM and do (on pinocchio use for instance TVM installed in `/opt/local/tvm/tvm-v0.16.0.rc0/`):
```
pip install -r tvm_requirements.txt
export PYTHONPATH=$PYTHONPATH:/path_to_tvm/python
```

For using jir backend, install JIR (ref to https://gitlab.inria/fr/jprotopo/jir.git) and set python path:
```
export PYTHONPATH=$PYTHONPATH:/path_to_jir
```

## Use it

+ Example in test.py
+ Just works with matmul (for now)

## Exploration

Use exploration script, for instance random 100 points for a simple matmul tiling strategy (3D tiling):

    ./explore.py --debug --search random --trials 100 --output results.random.csv

Use exploration script, for instance on input data generated on some tvm search (3D tiling + permutations), 2054 points here:

    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output data/results.mm06-tile4d.csv
    ...
    2054/2054 [55:54,  1.63s/it]
    real 3356 secs

With TVM:

    tile -p ./explore.py --debug --backend tvm --dims 256 256 512 --strategy tile4d --search data --data data/tvm_results.mm06.csv --output data/results.mm06-tile4d-tvm.csv
    ...
    2054/2054 [4:55,  8.65s/it]
    real 17765 secs

Use exhaustive search on a tiling strategy limited to tile4d + only vectorized tilings (450 points):

    # TVM backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend tvm --output data/results.mm06-tile4dv-tvm.csv
    450/450 [24:04,  3.21s/it]
    real 1444.50

    # MLIR backend, with default evaluation (sharedlib compilation and evaluate runtime with 5 repeats and 100 msecs min execution)
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend mlir --output data/results.mm06-tile4dv-mlir.csv
    450/450 [22:34<00:00,  3.01s/it]
    real 1355.98

    # XDSL backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend xdsl --output data/results.mm06-tile4dv-xdsl.csv
    450/450 [38:44<00:00,  5.17s/it]
    real 2329.94

    # JIR backend
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend jir --output data/results.mm06-tile4dv-xdsl.csv
    450/450 [22:30<00:00,  3.00s/it]
    real 1352.37

    # MLIR backend (jit compilation)
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend mlir --eval jit --output data/results.mm06-tile4dv-mlir.jit.csv
    450/450 [11:57<00:00,  1.59s/it]
    real 719.11

    # MLIR backend (static compilation)
    time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backend mlir --eval exe --output data/results.mm06-tile4dv-mlir.exe.csv
    450/450 [12:27<00:00,  1.66s/it]
    real 749.69


Test a single tiling with mlir backend and tvm backend:

    # Dumps and execute MLIR tiling
    ./explore.py --dump --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 1.89 msecs, peak perf: 26.38%
    # Dumps and execute TVM tiling
    ./explore.py --backend tvm --dump --debug --dims 256 256 512 --strategy tile4d --test 4 64 8 4
    ...
    INFO:__main__:Schedule: [4, 64, 8, 4]: time: 0.61 msecs, peak perf: 82.08%

Experiment on tile7d tiling strategy with 5000 points:

    time -p ./explore.py --debug --dims 256 256 512 --strategy tile7d --search random --trials 5000 --output results.mm06-tile7d-5000.csv
    ...
    5000/5000 [2:05:48,  1.51s/it]
    real 7550 secs


## Display

Result of exploration and display in `data/mlir_results.mm06-tile4d-all.svg` were generated with:

    ./explore.py --debug --dims 256 256 512 --strategy tile4d --search exhaustive --backend mlir --output data/results.mm06-tile4d.csv
    ./explore.py --debug --dims 256 256 512 --strategy tile4d --search exhaustive --backend tvm --output data/results.mm06-tile4d-tvm.csv
    ./display-results.py --output data/results.mm06-tile4d-all.svg --title "Exhaustive 1-level tiling + reorder (i,j,k, order) of 256x256x512 matmul" data/results.mm06-tile4d-tvm.csv:tvm:X:peak data/results.mm06-tile4d.csv:mlir:X:peak

Comparative performance distribution on tile4dv tilings for mlir and tvm backends in `data/mlir_results.mm06-tile4dv-all.svg` were generated with:

    ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search exhaustive --backends mlir tvm --output data/results.mm06-tile4dv-all.csv
    ./display-results.py  --output data/results.mm06-tile4dv-all.svg --title "Exhaustive 1-level tiling + reorder (i,j,k, order) of 256x256x512 vectorized matmul" data/results.mm06-tile4dv-all.csv:tvm:X:peak:tvm data/results.mm06-tile4dv-all.csv:mlir:X:peak:mlir

## Compile time comparison

Comparison of compile time (+1 execution actually) for the different compilation flows:

    # JIT:
    /usr/bin/time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search random --trials 10 --backend mlir --eval jit --output results.mm06-tile4dv-mlir-10.jit.csv
    10/10 [00:14<00:00,  1.47s/it]
    real 16.56
    # Compile exe and exec
    /usr/bin/time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search random --trials 10 --backend mlir --eval exe --output results.mm06-tile4dv-mlir-10.exe.csv
    10/10 [00:15<00:00,  1.52s/it]
    real 17.12
    # Compile shlib and eval runtime (force 1 run only)
    /usr/bin/time -p ./explore.py --debug --dims 256 256 512 --strategy tile4dv --search random --trials 10 --backend mlir --eval eval --repeat 1 --min-repeat-ms 0 --output results.mm06-tile4dv-mlir-10.eval.csv
    10/10 [00:14<00:00,  1.46s/it]
    real 16.56


## Notes

### Compile & link

After compilation using Python:
```
~/bin/llvm-xdsl/bin/clang /tmp/dump.bc ~/bin/llvm-xdsl/lib/libmlir_c_runner_utils.so -o /tmp/test
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpompougnac/bin/llvm-xdsl/lib/
/tmp/test
```

### Scalar FMAs

The option ```--math-uplift-to-fma``` combines arith operations into ```math.fma``` if the flag ```fast``` is set, but how to generate the latter ?
