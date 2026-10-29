#include "serial_spmv.h"
#include <stdexcept>

std::vector<double> spmv_serial(const SparseMatrix &matrix, const std::vector<double> &vector) {

    if (matrix.getCols() != static_cast<int>(vector.size())) {
        throw std::invalid_argument(
            "Matrix columns must match vector length for multiplication");
    }

    int rows = matrix.getRows();
    const auto &values = matrix.getValues();
    const auto &col_indices = matrix.getColIndices();
    const auto &row_ptr = matrix.getRowPtr();

    std::vector<double> result(rows, 0.0);

    // Perform matrix-vector multiplication, row by row
    for (int i = 0; i < rows; ++i) {
        // Get the range of non-zero elements in row i
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
