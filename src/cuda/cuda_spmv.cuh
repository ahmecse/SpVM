#ifndef CUDA_SPMV_CUH
#define CUDA_SPMV_CUH

#include "SparseMatrix.h"
#include <vector>

/**
 * Implements GPU-accelerated parallel algorithm:
 * - Each thread computes one row of the result
 * - Data transferred to GPU memory
 * - Kernel executed on GPU
 * - Results transferred back to CPU
 *
 * @param matrix 
 * @param vector 
 * @param blockSize (default: 256)
 */
std::vector<double> spmv_cuda(const SparseMatrix &matrix, const std::vector<double> &vector, int blockSize = 256);
bool cuda_available();
std::string get_cuda_device_info();

#endif // CUDA_SPMV_CUH
