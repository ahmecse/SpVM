#ifndef SERIAL_SPMV_H
#define SERIAL_SPMV_H

#include "SparseMatrix.h"
#include <vector>

/**
 * @param matrix Sparse matrix in CSR format
 * @param vector Dense vector to multiply with
 * @return Result vector of the multiplication
 */
std::vector<double> spmv_serial(const SparseMatrix &matrix, const std::vector<double> &vector);

#endif // SERIAL_SPMV_H
