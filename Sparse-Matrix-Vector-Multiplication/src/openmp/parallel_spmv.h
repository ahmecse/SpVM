/**
 * @file parallel_spmv.h
 * @brief Parallel implementation of Sparse Matrix-Vector Multiplication using OpenMP
 */

#ifndef PARALLEL_SPMV_H
#define PARALLEL_SPMV_H

#include "SparseMatrix.h"
#include <vector>

/**
 * @brief Parallel Sparse Matrix-Vector Multiplication using OpenMP
 *
 * Implements parallel algorithm with row-wise parallelization:
 * - Distributes rows among threads
 * - Each thread computes its assigned rows independently
 * - No synchronization needed (no race conditions)
 *
 * @param matrix Sparse matrix in CSR format
 * @param vector Dense vector to multiply with
 * @param num_threads Number of OpenMP threads (0 = use default)
 * @return Result vector of the multiplication
 */
std::vector<double> spmv_parallel(const SparseMatrix &matrix,
                                  const std::vector<double> &vector,
                                  int num_threads = 0);

#endif // PARALLEL_SPMV_H
