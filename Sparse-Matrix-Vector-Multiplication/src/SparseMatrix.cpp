#include "SparseMatrix.h"
#include <iomanip>

SparseMatrix::SparseMatrix(const std::vector<double> &vals, const std::vector<int> &col_idx, const std::vector<int> &row_p, int n_rows, int n_cols)
: values(vals), col_indices(col_idx), row_ptr(row_p), rows(n_rows), cols(n_cols), nnz(vals.size()) {

    // Validate input
    if (row_ptr.size() != static_cast<size_t>(rows + 1)) {
        throw std::invalid_argument("row_ptr size must be rows + 1");
    }

    if (col_indices.size() != values.size()) {
        throw std::invalid_argument("col_indices and values must have same size");
    }
}

// creates a sparse matrix in CSR format from a dense 2D array
SparseMatrix SparseMatrix::fromDense(const std::vector<std::vector<double>> &dense) {
    if (dense.empty() || dense[0].empty()) {
        throw std::invalid_argument("Matrix cannot be empty");
    }

    int n_rows = dense.size();
    int n_cols = dense[0].size();

    std::vector<double> vals;
    std::vector<int> col_idx;
    std::vector<int> row_p;
    row_p.push_back(0);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            if (dense[i][j] != 0.0) {
                vals.push_back(dense[i][j]);
                col_idx.push_back(j);
            }
        }
        row_p.push_back(vals.size());
    }
    return SparseMatrix(vals, col_idx, row_p, n_rows, n_cols);
}

// random sparse matrix generator for testing and benchmarking
SparseMatrix SparseMatrix::generateRandom(int n_rows, int n_cols, double density, unsigned int seed) {
    // density controls sparsity (0 = all zeros, 1 = dense)
    if (density < 0.0 || density > 1.0) {
        throw std::invalid_argument("Invalid! Density must be between 0 and 1");
    }

    std::mt19937 gen(seed); // Standard mersenne_twister_engine 
    std::uniform_real_distribution<> value_dist(0.0, 1.0);
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);

    std::vector<double> vals;
    std::vector<int> col_idx;
    std::vector<int> row_p;
    row_p.push_back(0);

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            if (prob_dist(gen) <= density) {
                vals.push_back(value_dist(gen));
                col_idx.push_back(j);
            }
        }
        row_p.push_back(vals.size());
    }
    return SparseMatrix(vals, col_idx, row_p, n_rows, n_cols);
}

// converts the sparse matrix in CSR format back to a dense 2D array
std::vector<std::vector<double>> SparseMatrix::toDense() const {
    std::vector<std::vector<double>> dense(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            dense[i][col_indices[j]] = values[j];
        }
    }
    return dense;
}

double SparseMatrix::getDensity() const {   // used in performance analysis
    if (rows == 0 || cols == 0)
        return 0.0;
    // Density = non-zeros / total elements
    return static_cast<double>(nnz) / (rows * cols);
}

void SparseMatrix::print() const {
    std::cout << "SparseMatrix(" << rows << "x" << cols << ", nnz=" << nnz << ", density=" << std::fixed << std::setprecision(4) << getDensity() * 100 << "%)" << std::endl;
}
