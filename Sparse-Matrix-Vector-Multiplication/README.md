# Parallel Sparse Matrix–Vector Multiplication in OpenMP & CUDA

### Workload Structure, Load Balance, and Data Movement

An OpenMP and CUDA study of workload structure, memory behavior, and overhead in
CSR sparse matrix–vector multiplication (SpMV), structured around four research
questions on scaling, load balance, memory behavior, and GPU data movement.

**Author:** Ahmed Shmels Muhe (22617089) · **Course:** Parallel Computing (3090440020) ·
**Instructor:** Prof. He Guo · **Dept. of Software Technology, Dalian University of Technology**
Contact: wzhang@mail.dlut.edu.cn

---

## 1. Overview

SpMV computes `y = A·x` for a sparse `m × n` matrix. Each nonzero costs one multiply
and one add (≈ `2·nnz` FLOPs), but each nonzero also moves 12 bytes of matrix data
(8-byte value + 4-byte index), so the kernel's arithmetic intensity is only
≈ **0.17 FLOP/byte** — *memory-bound*. Three properties make it hard to speed up:

1. **Low arithmetic intensity** — memory traffic, not FLOPs, sets the pace.
2. **Irregular access to `x`** — `col_idx` scatters reads across the vector.
3. **Uneven row lengths** — equal row counts can mean unequal nonzero work per thread.

This repository implements and benchmarks five CSR SpMV kernels, covering RQ1–RQ4:

| Tag | Kernel | Status |
|-----|--------|--------|
| **S0** | Serial CSR row loop (baseline) | implemented, benchmarked |
| **O0** | OpenMP static row-parallel loop | implemented, benchmarked (2 threads) |
| **G0** | CUDA scalar kernel, one thread per row | implemented, correctness-validated |
| **G1** | CUDA shared-memory kernel (stages `x` when `8n ≤ 48` KiB, i.e. `n ≤ 6144`) | implemented; auto-selected by size |
| — | Nonzero-aware row partitioning (`bounds[t]` via prefix counts) | designed; not yet benchmarked |

### Research questions

- **RQ1 (Scaling):** How does OpenMP SpMV scale with threads on fixed and growing workloads?
- **RQ2 (Load balance):** When does static row scheduling fail, and does nonzero-aware partitioning help?
- **RQ3 (Memory behavior):** How do matrix footprint and irregular access to `x` limit performance?
- **RQ4 (GPU data movement):** When does CUDA kernel speedup survive allocation and host–device transfers?

Timing scope: kernel time (`T_kernel`, CUDA events) is reported separately from
end-to-end time (`T_E2E` = setup + H2D + kernel + D2H + cleanup).

## 2. Repository structure

```
├── src/
│   ├── SparseMatrix.{h,cpp}      # CSR matrix class (values, col_indices, row_ptr)
│   ├── serial/                   # S0 serial kernel
│   ├── openmp/                   # O0 OpenMP kernel
│   ├── cuda/                     # G0 scalar + G1 shared-memory kernels
│   ├── testing/                  # validate(.cuda).cpp, benchmark(.cuda).cpp
│   └── plotting/                 # plot_results.py, plot_cuda_results.py
├── docs/
│   └── CUDA_GUIDE.md             # CUDA build/run guide
├── Makefile                      # CPU build (serial + OpenMP)
├── Makefile.cuda                 # CUDA build (adds G0/G1)
├── benchmark_results.csv         # Recorded two-thread pilot (7 sizes, seed 42)
└── README.md
```

## 3. Build and run

### CPU (serial + OpenMP)

```bash
make all        # build validate and benchmark
make test       # correctness: 4×4, 100×100 @10%, 200×200 @2%, diagonal
make bench      # writes benchmark_results.csv and results/*.png
make plot       # speedup / execution-time / efficiency figures
make run        # full workflow
```

Requirements: GCC ≥ 7 with C++17 and OpenMP; Python 3 with matplotlib/pandas for plots.
Thread count: `export OMP_NUM_THREADS=<p>` or edit `benchmark.cpp`.

### CUDA (optional)

```bash
make -f Makefile.cuda all    # requires CUDA toolkit (targets sm_60; adjust -arch for your GPU)
make -f Makefile.cuda test   # includes CUDA validation (e.g. 1000×1000 @5%)
make -f Makefile.cuda bench  # serial vs OpenMP vs CUDA comparison
```

## 4. Validation

Every kernel is checked against dense matrix multiplication ground truth,
`ε_max = max_i |y_impl − y_truth|`. All recorded runs satisfy `ε_max < 1e-10`.
Row-wise parallelism is race-free by construction (each thread owns distinct `y[i]`).

## 5. Recorded results (two-thread CPU pilot)

Random square matrices at nominal 10% density, seed 42; whole-call means of 5 timed
runs after 2 warm-ups. **This is a size sweep, not a scaling study** — a 2-thread
point cannot establish strong/weak scaling; see §6.

| n | nnz | Serial (ms) | OpenMP (ms) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 100 | 1,002 | 0.001458 | 0.001726 | 0.845× | 42.2% |
| 200 | 3,957 | 0.005164 | 0.004930 | 1.047× | 52.4% |
| 500 | 25,008 | 0.031886 | 0.026172 | 1.218× | 60.9% |
| 1,000 | 100,084 | 0.118434 | 0.113091 | 1.047× | 52.4% |
| 2,000 | 400,247 | 0.550202 | 0.420618 | 1.308× | 65.4% |
| 5,000 | 2,498,138 | 3.477790 | 2.436810 | **1.427×** | **71.4%** |
| 10,000 | 9,999,375 | 14.258790 | 10.178649 | 1.401× | 70.0% |

(Speedups recomputed as the ratio of the displayed means.)

Reading:

- **n = 100 is a slowdown (0.845×)** — thread startup + barrier dominate tiny workloads.
- From **n = 2,000 onward** the speedup is a consistent 1.31–1.43× — the signature of a
  bandwidth-limited kernel on 2 cores, not a compute-limited one.
- Derived throughput at n = 10,000: 1.40 → 1.96 GFLOP/s, effective algorithmic bandwidth
  8.4 → 11.8 GB/s — consistent with ~0.17 FLOP/byte intensity.

## 6. Roadmap (experiments required to close RQ1–RQ4)

- **Scaling sweep:** `OMP_NUM_THREADS = 1,2,4,8,…` per fixed matrix → strong scaling;
  workloads with `nnz_p/p ≈ const` (n grows as √p) → weak scaling.
- **Scheduling ablation:** static vs dynamic/guided vs nonzero-aware partitioning on
  regular *and* skewed matrices, reporting work- and time-imbalance separately.
- **CUDA end-to-end timing:** per-repetition `T_kernel` (events) *and* directly measured
  `T_E2E` (wall clock), never summed component medians.
- **Workload breadth:** sparser densities, severe row-skew, and real SuiteSparse matrices.

## 7. References

1. N. Bell and M. Garland, "Implementing sparse matrix-vector multiplication on
   throughput-oriented processors," SC '09. doi:10.1145/1654059.1654078
2. S. Filippone et al., "Sparse matrix-vector multiplication on GPGPUs,"
   ACM TOMS 43(4), 2017. doi:10.1145/3017994
3. S. Williams et al., "Optimization of sparse matrix-vector multiplication on emerging
   multicore platforms," Parallel Computing 35(3), 2009. doi:10.1016/j.parco.2008.12.006
4. Y. Liu and B. Schmidt, "LightSpMV: Faster CUDA-compatible sparse matrix-vector
   multiplication using compressed sparse rows," J. Signal Process. Syst. 90, 2018.
   doi:10.1007/s11265-017-1280-7
5. G. He and J. Gao, "A novel CSR-based sparse matrix-vector multiplication on GPUs,"
   Math. Probl. Eng., 2016. doi:10.1155/2016/8471283
6. A. Buluç et al., "Parallel sparse matrix-vector and matrix-transpose-vector
   multiplication using compressed sparse blocks," SPAA '09. doi:10.1145/1583991.1584053
7. C. Giannoula et al., "SparseP: Towards efficient sparse matrix vector multiplication on
   real processing-in-memory architectures," PACM Meas. Anal. Comput. Syst. 6(1), 2022.
   doi:10.1145/3508041
