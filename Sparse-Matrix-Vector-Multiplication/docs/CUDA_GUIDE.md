# CUDA Implementation Guide

## Overview

This directory now includes a **CUDA-based GPU-accelerated** implementation of Sparse Matrix-Vector Multiplication (SpMV) in addition to the serial and OpenMP parallel versions.

## 🎯 CUDA Implementation Features

### Architecture
- **Kernel Design**: One thread per matrix row
- **Memory Management**: Efficient host-device transfers
- **Optimization**: Shared memory caching for small vectors
- **Error Handling**: Comprehensive CUDA error checking

### Two Kernel Variants

1. **Basic Kernel** (`spmv_csr_kernel`)
   - Simple and efficient for most cases
   - Each thread computes one row
   - Direct global memory access

2. **Optimized Kernel** (`spmv_csr_shared_kernel`)
   - Uses shared memory for vector caching
   - Activated for small vectors (< 48KB)
   - Reduces global memory bandwidth

## 📋 Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 6.0 or higher
- Minimum 2GB GPU memory recommended

### Software
- CUDA Toolkit 11.0 or higher
- NVIDIA GPU drivers
- GCC 7.0+ or compatible host compiler

### Installation

**Ubuntu/Debian:**
```bash
# Install CUDA Toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvcc --version
nvidia-smi
```

**Check GPU availability:**
```bash
nvidia-smi
```

## 🚀 Building and Running

### Build CUDA Programs

Use the CUDA-specific Makefile:

```bash
# Build all CUDA programs
make -f Makefile.cuda all

# Check CUDA availability first
make -f Makefile.cuda check_cuda

# Build only validation
make -f Makefile.cuda validate_cuda

# Build only benchmark
make -f Makefile.cuda benchmark_cuda
```

### Run Validation

```bash
# Build and test
make -f Makefile.cuda test

# Or run directly
./validate_cuda
```

Expected output:
```
CUDA Device Information:
  Device: [Your GPU Name], Compute Capability: X.X, ...

Test Case 1: Small Predefined Matrix
Serial: ✓ (0.00e+00)  OpenMP: ✓ (0.00e+00)  CUDA: ✓ (0.00e+00)
...
✓ ALL TESTS PASSED
```

### Run Benchmarks

```bash
# Build and benchmark
make -f Makefile.cuda bench

# Or run directly
./benchmark_cuda
```

### Generate Performance Graphs

```bash
# Install Python dependencies (if not already installed)
pip3 install matplotlib pandas numpy

# Generate graphs
make -f Makefile.cuda plot

# Or run directly
python3 plot_cuda_results.py
```

This creates 5 graphs in `results_cuda/`:
1. `execution_time_cuda.png` - Serial vs OpenMP vs CUDA
2. `speedup_comparison.png` - OpenMP and CUDA speedups
3. `performance_bars.png` - Bar chart comparison
4. `execution_time_log.png` - Log-scale comparison
5. `cuda_vs_openmp.png` - CUDA vs OpenMP relative performance

### Complete Workflow

```bash
make -f Makefile.cuda run
```

This executes: check_cuda → build → test → benchmark → plot

## 📁 CUDA Files

| File | Description |
|------|-------------|
| `cuda_spmv.cuh` | CUDA SpMV header file |
| `cuda_spmv.cu` | CUDA kernel implementation |
| `validate_cuda.cpp` | Validation with CUDA tests |
| `benchmark_cuda.cpp` | Performance benchmarking |
| `Makefile.cuda` | Build system for CUDA |
| `plot_cuda_results.py` | Visualization script |
| `CUDA_GUIDE.md` | This file |

## 🔬 Algorithm Details

### CUDA Kernel Pseudocode

```cuda
__global__ void spmv_csr_kernel(...) {
    row = blockIdx.x * blockDim.x + threadIdx.x
    
    if (row < num_rows) {
        sum = 0.0
        for j = row_ptr[row] to row_ptr[row+1]:
            col = col_indices[j]
            sum += values[j] * vector[col]
        
        result[row] = sum
    }
}
```

### Memory Transfers

```
CPU → GPU:
  - Matrix values (nnz doubles)
  - Column indices (nnz ints)
  - Row pointers ((rows+1) ints)
  - Input vector (cols doubles)

GPU → CPU:
  - Result vector (rows doubles)
```

### Parallelization Strategy

- **Partitioning**: Each row is an independent task
- **Communication**: One-time data transfer (host ↔ device)
- **Synchronization**: Implicit barrier after kernel launch
- **Load Balancing**: Static assignment (one thread per row)

## 📊 Performance Characteristics

### When CUDA Excels
- ✅ Large matrices (10,000+ rows)
- ✅ Moderate to high density (5-20%)
- ✅ Many non-zeros per row
- ✅ Repeated SpMV operations (amortize transfer cost)

### When OpenMP May Be Better
- ⚠️ Very small matrices (< 1000 rows)
- ⚠️ Extremely sparse matrices (< 1% density)
- ⚠️ Single SpMV operation (transfer overhead)

### Typical Speedups

Based on typical results (your mileage may vary):

| Matrix Size | Serial | OpenMP | CUDA |
|-------------|--------|--------|------|
| 1,000×1,000 | 1.0x | 3-5x | 5-10x |
| 10,000×10,000 | 1.0x | 4-8x | 20-50x |
| 50,000×50,000 | 1.0x | 5-10x | 50-100x |

## 🔧 Tuning Parameters

### Block Size
Default: 256 threads per block

Modify in code:
```cpp
auto result = spmv_cuda(matrix, vector, 512);  // Use 512 threads
```

Common values: 128, 256, 512, 1024

### Benchmark Configuration

Edit `benchmark_cuda.cpp`:
```cpp
std::vector<int> sizes = {100, 500, 1000, 5000, 10000, 50000, 100000};
double density = 0.05;  // 5% density
```

## 🐛 Troubleshooting

### CUDA Not Found
```bash
# Check if CUDA is installed
nvcc --version

# If not, install CUDA Toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads
```

### No GPU Detected
```bash
# Check GPU status
nvidia-smi

# If no output, check drivers:
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Compilation Errors

**Error: `nvcc: command not found`**
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: `unsupported GNU version`**
```bash
# Use compatible GCC version
export CXX=g++-9
make -f Makefile.cuda all
```

### Runtime Errors

**Error: `out of memory`**
- Reduce matrix size
- Use smaller batch sizes
- Free GPU memory: `nvidia-smi` then kill processes

**Error: `invalid device function`**
- Update GPU drivers
- Check compute capability matches
- Modify `-arch` flag in Makefile.cuda

## 📈 Understanding Results

### Speedup Analysis

**Good CUDA speedup** (20-100x):
- Large problem size
- Efficient memory access
- Good GPU utilization

**Low CUDA speedup** (1-5x):
- Small problem size → overhead dominates
- Sparse data → underutilization
- Memory bandwidth bottleneck

### Profiling (Optional)

```bash
# Profile CUDA kernel
nvprof ./benchmark_cuda

# Or use NVIDIA Nsight
nsys profile ./benchmark_cuda
```

## 📝 For Your Report

### Additional Sections for CUDA

1. **CUDA Algorithm Design**
   - Thread-to-row mapping
   - Memory transfer strategy
   - Kernel optimization (shared memory)

2. **Performance Analysis**
   - Compare serial vs OpenMP vs CUDA
   - Analyze speedup trends
   - Discuss when GPU excels

3. **Memory Bandwidth**
   - Calculate theoretical vs achieved bandwidth
   - Discuss memory bottlenecks

4. **Challenges**
   - Data transfer overhead
   - Load balancing for irregular sparsity
   - Memory coalescing

### Sample Report Content

```
III. CUDA IMPLEMENTATION

A. GPU Architecture
   - NVIDIA GPU with X SMs
   - Y CUDA cores per SM
   - Z GB global memory

B. Kernel Design
   - One thread per matrix row
   - CSR format preserved on GPU
   - Shared memory optimization

C. Performance Results
   - Average CUDA speedup: XX.Xx
   - Best case: YY.Yx on ZZ,ZZZ matrix
   - CUDA vs OpenMP: W.Wx faster
```

## 🎓 Learning Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Sparse Matrix Algorithms on GPUs](https://developer.nvidia.com/cusparse)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/nsight-systems)

## ⚡ Quick Reference

```bash
# Build everything
make -f Makefile.cuda all

# Test correctness
./validate_cuda

# Benchmark performance
./benchmark_cuda

# Generate graphs
python3 plot_cuda_results.py

# Clean up
make -f Makefile.cuda clean
```

## 🎉 Summary

You now have **three implementations**:
1. ✅ **Serial** - Baseline performance
2. ✅ **OpenMP** - CPU multi-core parallelism
3. ✅ **CUDA** - GPU-accelerated computation

All three are validated, benchmarked, and ready for your project submission!
