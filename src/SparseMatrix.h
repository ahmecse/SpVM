#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

class SparseMatrix {
private:
    std::vector<double> values;   // Non-zero values
    std::vector<int> col_indices; // Col indices of non-zero values
    std::vector<int> row_ptr;     // Row pointers
    int rows;                     // Num of rows
    int cols;                     // Num of cols
    int nnz;                      // Num of non-zero elements

public:
    /**
     * @param vals 
     * @param col_idx 
     * @param row_p 
     * @param n_rows 
     * @param n_cols 
     */
    SparseMatrix(const std::vector<double> &vals, const std::vector<int> &col_idx, const std::vector<int> &row_p, int n_rows, int n_cols);

    /**
     * @param dense 
     * @return 
     */
    static SparseMatrix fromDense(const std::vector<std::vector<double>> &dense);

    /**
     * @param n_rows 
     * @param n_cols 
     * @param density 
     * @param seed 
     * @return 
     */
    static SparseMatrix generateRandom(int n_rows, int n_cols, double density = 0.1, unsigned int seed = 42);

    /**
     * @return Dense matrix as 2D vector
     */
    std::vector<std::vector<double>> toDense() const;

    // Getters
    const std::vector<double> &getValues() const { return values; }
    const std::vector<int> &getColIndices() const { return col_indices; }
    const std::vector<int> &getRowPtr() const { return row_ptr; }
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    int getNnz() const { return nnz; }

    /**
     * @return Fraction of non-zero elements
     */
    double getDensity() const;

    void print() const;
};

#endif // SPARSE_MATRIX_H
