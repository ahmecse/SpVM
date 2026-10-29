#include "parallel_spmv.h"
#include <stdexcept>
#include <omp.h>

// Function to perform parallel sparse matrix-vector multiplication using OpenMP
// Each row of the matrix is processed in parallel to compute the resulting vector
// Each thread processes different rows independently
std::vector<double> spmv_parallel(const SparseMatrix &matrix, const std::vector<double> &vector, int num_threads) {
    if (matrix.getCols() != static_cast<int>(vector.size())) {
        throw std::invalid_argument("Matrix cols must match vector length for multiplication");
    }

    int rows = matrix.getRows();
    const auto &values = matrix.getValues();
    const auto &col_indices = matrix.getColIndices();
    const auto &row_ptr = matrix.getRowPtr();

    // one output element per row, each thread writes to unique index
    std::vector<double> result(rows, 0.0);

    // Set num of threads
    // if num_threads <= 0, OpenMP uses default
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

// Each iteration is independent (no race conditions)
// static scheduling for balanced workload distribution
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        double sum = 0.0;
        for (int j = row_start; j < row_end; ++j) {
            int col = col_indices[j];
            double value = values[j];
            sum += value * vector[col];
        }
        result[i] = sum;
    }

    return result;
}
