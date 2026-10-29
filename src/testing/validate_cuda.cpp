#include "SparseMatrix.h"
#include "serial_spmv.h"
#include "parallel_spmv.h"
#include "cuda_spmv.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

// Compute dense matrix-vector multiplication (ground truth)
std::vector<double> dense_matvec(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

// Check if two vectors are approximately equal
bool vectors_equal(const std::vector<double> &v1, const std::vector<double> &v2, double tolerance = 1e-10) {
    if (v1.size() != v2.size())
        return false;

    for (size_t i = 0; i < v1.size(); ++i)  {
        if (std::abs(v1[i] - v2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Compute maximum absolute error
double max_error(const std::vector<double> &v1, const std::vector<double> &v2) {
    double max_err = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double err = std::abs(v1[i] - v2[i]);
        if (err > max_err)
            max_err = err;
    }
    return max_err;
}

void print_test_result(const std::string &test_name, bool serial_correct, bool parallel_correct, bool cuda_correct, double serial_error, double parallel_error, double cuda_error) {
    std::cout << std::left << std::setw(30) << test_name;
    std::cout << "Serial: " << (serial_correct ? "✓" : "✗") << " ("
              << std::scientific << std::setprecision(2) << serial_error << ")  ";
    std::cout << "OpenMP: " << (parallel_correct ? "✓" : "✗") << " ("
              << std::scientific << std::setprecision(2) << parallel_error << ")  ";
    std::cout << "CUDA: " << (cuda_correct ? "✓" : "✗") << " ("
              << std::scientific << std::setprecision(2) << cuda_error << ")"
              << std::endl;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  SPARSE MATRIX-VECTOR MULTIPLICATION VALIDATION (with CUDA)\n";
    std::cout << "======================================================================\n\n";

    if (!cuda_available()) {
        std::cerr << "ERROR: No CUDA device available!\n";
        std::cerr << "Please ensure CUDA is properly installed and a GPU is present.\n";
        return 1;
    }

    std::cout << "CUDA Device Information:\n";
    std::cout << "  " << get_cuda_device_info() << "\n\n";

    bool all_passed = true;

    // Test Case 1: Small predefined matrix
    std::cout << "Test Case 1: Small Predefined Matrix\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::vector<std::vector<double>> dense1 = {
        {1.0, 0.0, 2.0, 0.0},
        {0.0, 3.0, 0.0, 4.0},
        {5.0, 0.0, 6.0, 0.0},
        {0.0, 0.0, 0.0, 7.0}};
    std::vector<double> vector1 = {1.0, 2.0, 3.0, 4.0};

    SparseMatrix sparse1 = SparseMatrix::fromDense(dense1);
    std::cout << "Matrix: ";
    sparse1.print();

    auto expected1 = dense_matvec(dense1, vector1);
    auto serial1 = spmv_serial(sparse1, vector1);
    auto parallel1 = spmv_parallel(sparse1, vector1);
    auto cuda1 = spmv_cuda(sparse1, vector1);

    bool s1 = vectors_equal(serial1, expected1);
    bool p1 = vectors_equal(parallel1, expected1);
    bool c1 = vectors_equal(cuda1, expected1);
    double e1_s = max_error(serial1, expected1);
    double e1_p = max_error(parallel1, expected1);
    double e1_c = max_error(cuda1, expected1);

    print_test_result("Small 4x4 Matrix", s1, p1, c1, e1_s, e1_p, e1_c);
    all_passed = all_passed && s1 && p1 && c1;

    // Test Case 2: Random sparse matrix
    std::cout << "\nTest Case 2: Random Sparse Matrix (100x100, 10% density)\n";
    std::cout << "----------------------------------------------------------------------\n";
    SparseMatrix sparse2 = SparseMatrix::generateRandom(100, 100, 0.1, 42);
    std::cout << "Matrix: ";
    sparse2.print();

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::vector<double> vector2(100);
    for (auto &v : vector2)
        v = dist(gen);

    auto dense2 = sparse2.toDense();
    auto expected2 = dense_matvec(dense2, vector2);
    auto serial2 = spmv_serial(sparse2, vector2);
    auto parallel2 = spmv_parallel(sparse2, vector2);
    auto cuda2 = spmv_cuda(sparse2, vector2);

    bool s2 = vectors_equal(serial2, expected2);
    bool p2 = vectors_equal(parallel2, expected2);
    bool c2 = vectors_equal(cuda2, expected2);
    double e2_s = max_error(serial2, expected2);
    double e2_p = max_error(parallel2, expected2);
    double e2_c = max_error(cuda2, expected2);

    print_test_result("Random 100x100", s2, p2, c2, e2_s, e2_p, e2_c);
    all_passed = all_passed && s2 && p2 && c2;

    // Test Case 3: Very sparse matrix
    std::cout << "\nTest Case 3: Very Sparse Matrix (200x200, 2% density)\n";
    std::cout << "----------------------------------------------------------------------\n";
    SparseMatrix sparse3 = SparseMatrix::generateRandom(200, 200, 0.02, 123);
    std::cout << "Matrix: ";
    sparse3.print();

    std::vector<double> vector3(200);
    for (auto &v : vector3)
        v = dist(gen);

    auto dense3 = sparse3.toDense();
    auto expected3 = dense_matvec(dense3, vector3);
    auto serial3 = spmv_serial(sparse3, vector3);
    auto parallel3 = spmv_parallel(sparse3, vector3);
    auto cuda3 = spmv_cuda(sparse3, vector3);

    bool s3 = vectors_equal(serial3, expected3);
    bool p3 = vectors_equal(parallel3, expected3);
    bool c3 = vectors_equal(cuda3, expected3);
    double e3_s = max_error(serial3, expected3);
    double e3_p = max_error(parallel3, expected3);
    double e3_c = max_error(cuda3, expected3);

    print_test_result("Very Sparse 200x200", s3, p3, c3, e3_s, e3_p, e3_c);
    all_passed = all_passed && s3 && p3 && c3;

    // Test Case 4: Larger matrix to showcase GPU
    std::cout << "\nTest Case 4: Larger Matrix (1000x1000, 5% density)\n";
    std::cout << "----------------------------------------------------------------------\n";
    SparseMatrix sparse4 = SparseMatrix::generateRandom(1000, 1000, 0.05, 999);
    std::cout << "Matrix: ";
    sparse4.print();

    std::vector<double> vector4(1000);
    for (auto &v : vector4)
        v = dist(gen);

    auto serial4 = spmv_serial(sparse4, vector4);
    auto parallel4 = spmv_parallel(sparse4, vector4);
    auto cuda4 = spmv_cuda(sparse4, vector4);

    // Use serial as reference for large matrix (dense conversion expensive)
    bool p4 = vectors_equal(parallel4, serial4);
    bool c4 = vectors_equal(cuda4, serial4);
    double e4_s = 0.0;
    double e4_p = max_error(parallel4, serial4);
    double e4_c = max_error(cuda4, serial4);

    print_test_result("Larger 1000x1000", true, p4, c4, e4_s, e4_p, e4_c);
    all_passed = all_passed && p4 && c4;

    std::cout << "\n======================================================================\n";
    std::cout << "                         VALIDATION SUMMARY\n";
    std::cout << "======================================================================\n";

    if (all_passed) {
        std::cout << "✓ ALL TESTS PASSED - All implementations are correct!\n";
    }
    else {
        std::cout << "✗ SOME TESTS FAILED - Please review implementations!\n";
    }
    std::cout << "======================================================================\n";
    return all_passed ? 0 : 1;
}
