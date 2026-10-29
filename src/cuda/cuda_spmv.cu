#include "cuda_spmv.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess)                                          \
        {                                                                  \
            std::stringstream ss;                                          \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
               << cudaGetErrorString(error);                               \
            throw std::runtime_error(ss.str());                            \
        }                                                                  \
    } while (0)

/**
 *
 * @param values
 * @param col_indices
 * @param row_ptr
 * @param vector
 * @param result 
 * @param num_rows 
 */

__global__ void spmv_csr_kernel(const double *values, const int *col_indices, const int *row_ptr, const double *vector, double *result, int num_rows) {
    // Calculate global thread ID
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        // dot product for this row
        for (int j = row_start; j < row_end; ++j) {
            int col = col_indices[j];
            sum += values[j] * vector[col];
        }
        result[row] = sum;
    }
}

// Optimized CUDA kernel using shared memory for vector caching
__global__ void spmv_csr_shared_kernel(const double *values, const int *col_indices, const int *row_ptr, const double *vector, double *result, int num_rows, int num_cols) {
    extern __shared__ double shared_vector[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // load vector into shared memory, each thread loads one element
    for (int i = tid; i < num_cols; i += blockDim.x) {
        shared_vector[i] = vector[i];
    }
    __syncthreads();

    if (row < num_rows) {
        double sum = 0.0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = col_indices[j];
            if (col < num_cols) {
                sum += values[j] * shared_vector[col];
            }
            else {
                sum += values[j] * vector[col];
            }
        }
        result[row] = sum;
    }
}

std::vector<double> spmv_cuda(const SparseMatrix &matrix, const std::vector<double> &vector, int blockSize) {
    if (matrix.getCols() != static_cast<int>(vector.size())) {
        throw std::invalid_argument("Matrix columns must match vector length for multiplication");
    }

    int rows = matrix.getRows();
    int cols = matrix.getCols();
    int nnz = matrix.getNnz();
    const auto &values = matrix.getValues();
    const auto &col_indices = matrix.getColIndices();
    const auto &row_ptr = matrix.getRowPtr();
    double *d_values, *d_vector, *d_result;
    int *d_col_indices, *d_row_ptr;

    //allocate device memory
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vector, cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, rows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector, vector.data(), cols * sizeof(double), cudaMemcpyHostToDevice));

    // grid dimensions
    int numBlocks = (rows + blockSize - 1) / blockSize;

    // launch kernel
    if (cols * sizeof(double) <= 48 * 1024) { 
        size_t sharedMemSize = cols * sizeof(double);
        spmv_csr_shared_kernel<<<numBlocks, blockSize, sharedMemSize>>>(d_values, d_col_indices, d_row_ptr, d_vector, d_result, rows, cols);
    }
    else {
        spmv_csr_kernel<<<numBlocks, blockSize>>>(d_values, d_col_indices, d_row_ptr, d_vector, d_result, rows);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<double> result(rows);
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, rows * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_result));

    return result;
}

bool cuda_available() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

std::string get_cuda_device_info() {
    if (!cuda_available()) {
        return "No CUDA device available";
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::stringstream ss;
    ss << "Device: " << prop.name
       << ", Compute Capability: " << prop.major << "." << prop.minor
       << ", Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB"
       << ", SMs: " << prop.multiProcessorCount;
    return ss.str();
}
