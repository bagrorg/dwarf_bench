# Dwarf Bench and HDK
## Dwarf Bench installation

1. Download release build from [here](https://github.com/kurapov-peter/dwarf_bench/releases). For example
```console
wget https://github.com/kurapov-peter/dwarf_bench/releases/download/v0.2.0/dbench-0.2.0-Release.tar.gz
```

2. Ensure that you have
+ SYCL
+ Boost (optional, needed only for `dwarf_bench` utility)
+ TBB

3. You also need to install OpenCL (for CPU) and CUDA (for GPU)

4. Add `dbench` to environment to make it visible for cmake:
```console
export dbench_DIR=/path/to/dbench/install/
```

### Installing SYCL
There are three known ways to install SYCL:
1. Build it from [source](https://github.com/intel/llvm)
2. Download it via package manager (ATTENTION! it will install heavy oneAPI kit)
3. Download it with Conda

### Installing OpenCL
For OpenCL installation follow this link -- https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html

## Building HDK 
To build HDK with Dwarfs simply run
```console
cmake -DENABLE_CUDA=on -DENABLE_DWARF_BENCH=on ..
make -j
```
