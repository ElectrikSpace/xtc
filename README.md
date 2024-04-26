# Xdsl Transform

## Installation instructions

### Install xdsl

```
git clone git@github.com:xdslproject/xdsl.git
cd xdsl
pip install -e .
```

### Install the right version of MLIR

```
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/bin/llvm-xdsl -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_SHARED_LINKER_FLAGS="-fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fno-omit-frame-pointer" ../llvm
make -j4
make install
```

## Use it

+ Example in test.py
+ Just works with matmul (for now)
