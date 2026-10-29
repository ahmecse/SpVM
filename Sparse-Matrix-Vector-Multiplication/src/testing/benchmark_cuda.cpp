#include "SparseMatrix.h"
#include "serial_spmv.h"
#include "parallel_spmv.h"
#include "cuda_spmv.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <omp.h>

struct BenchmarkResult
{
    int size;
    int nnz;
    double density;
    double serial_time;
    double openmp_time;
    double cuda_time;
    double openmp_speedup;
    double cuda_speedup;
    double openmp_efficiency;
    int num_threads;
};

// Measure execution time for a function
template <typename Func>
double measure_time(Func func, int warmup = 2, int runs = 5) {
    // Warmup runs
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // Timed runs
    std::vector<double> times;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        times.push_back(diff.count());
    }

    double total = 0.0;
    for (double t : times)
        total += t;
    return total / times.size();
}

// Benchmark a single matrix size
BenchmarkResult benchmark_size(int size, double density, int num_threads) {
    SparseMatrix matrix = SparseMatrix::generateRandom(size, size, density, 42);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::vector<double> vector(size);
    for (auto &v : vector)
        v = dist(gen);

    // Measure serial time
    double serial_time = measure_time([&]()
                                      { spmv_serial(matrix, vector); });

    // Measure OpenMP parallel time
    double openmp_time = measure_time([&]()
                                      { spmv_parallel(matrix, vector, num_threads); });

    // Measure CUDA time
    double cuda_time = measure_time([&]()
                                    { spmv_cuda(matrix, vector); });

    // Calculate speedups and efficiency
    double openmp_speedup = serial_time / openmp_time;
    double cuda_speedup = serial_time / cuda_time;
    double openmp_efficiency = openmp_speedup / num_threads;

    BenchmarkResult result;
    result.size = size;
    result.nnz = matrix.getNnz();
    result.density = matrix.getDensity();
    result.serial_time = serial_time;
    result.openmp_time = openmp_time;
    result.cuda_time = cuda_time;
    result.openmp_speedup = openmp_speedup;
    result.cuda_speedup = cuda_speedup;
    result.openmp_efficiency = openmp_efficiency;
    result.num_threads = num_threads;

    return result;
}

void save_results_csv(const std::vector<BenchmarkResult> &results, const std::string &filename) {
    std::ofstream file(filename);

    file << "size,nnz,density,serial_time_ms,openmp_time_ms,cuda_time_ms,"
         << "openmp_speedup,cuda_speedup,openmp_efficiency,num_threads\n";

    for (const auto &r : results) {
        file << r.size << ","
             << r.nnz << ","
             << std::fixed << std::setprecision(6) << r.density << ","
             << r.serial_time * 1000 << ","
             << r.openmp_time * 1000 << ","
             << r.cuda_time * 1000 << ","
             << r.openmp_speedup << ","
             << r.cuda_speedup << ","
             << r.openmp_efficiency << ","
             << r.num_threads << "\n";
    }
    file.close();
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  SPARSE MATRIX-VECTOR MULTIPLICATION BENCHMARK (with CUDA)\n";
    std::cout << "======================================================================\n";

    if (!cuda_available()) {
        std::cerr << "ERROR: No CUDA device available!\n";
        std::cerr << "This benchmark requires a CUDA-capable GPU.\n";
        return 1;
    }

    std::cout << "\nCUDA Device Information:\n";
    std::cout << "  " << get_cuda_device_info() << "\n\n";

    std::vector<int> sizes = {100, 500, 1000, 2000, 5000, 10000, 20000, 50000};
    double density = 0.1;
    int num_threads = omp_get_max_threads();

    std::cout << "Configuration:\n";
    std::cout << "  Density: " << density * 100 << "%\n";
    std::cout << "  OpenMP Threads: " << num_threads << "\n";
    std::cout << "  Matrix Sizes: ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i < sizes.size() - 1)
            std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "======================================================================\n\n";

    std::vector<BenchmarkResult> results;

    for (int size : sizes) {
        std::cout << "Testing " << size << "x" << size << " matrix...\n";

        BenchmarkResult result = benchmark_size(size, density, num_threads);
        results.push_back(result);

        std::cout << "  Non-zeros:      " << result.nnz << "\n";
        std::cout << "  Serial Time:    " << std::fixed << std::setprecision(3)
                  << result.serial_time * 1000 << " ms\n";
        std::cout << "  OpenMP Time:    " << result.openmp_time * 1000 << " ms\n";
        std::cout << "  CUDA Time:      " << result.cuda_time * 1000 << " ms\n";
        std::cout << "  OpenMP Speedup: " << std::setprecision(2)
                  << result.openmp_speedup << "x\n";
        std::cout << "  CUDA Speedup:   " << result.cuda_speedup << "x\n";
        std::cout << "  Best Method:    " << (result.cuda_time < result.openmp_time ? "CUDA" : "OpenMP") << "\n\n";
    }

    std::cout << "======================================================================\n";
    std::cout << "                    BENCHMARK RESULTS SUMMARY\n";
    std::cout << "======================================================================\n";
    std::cout << std::left
              << std::setw(8) << "Size"
              << std::setw(12) << "NNZ"
              << std::setw(12) << "Serial(ms)"
              << std::setw(12) << "OpenMP(ms)"
              << std::setw(12) << "CUDA(ms)"
              << std::setw(12) << "OMP_Speedup"
              << std::setw(12) << "CUDA_Speedup" << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (const auto &r : results) {
        std::cout << std::left
                  << std::setw(8) << r.size
                  << std::setw(12) << r.nnz
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.serial_time * 1000
                  << std::setw(12) << r.openmp_time * 1000
                  << std::setw(12) << r.cuda_time * 1000
                  << std::setprecision(2)
                  << std::setw(12) << r.openmp_speedup
                  << std::setw(12) << r.cuda_speedup << "\n";
    }
    std::cout << "======================================================================\n";

    double total_openmp_speedup = 0.0;
    double total_cuda_speedup = 0.0;
    double max_openmp_speedup = 0.0;
    double max_cuda_speedup = 0.0;

    for (const auto &r : results) {
        total_openmp_speedup += r.openmp_speedup;
        total_cuda_speedup += r.cuda_speedup;
        if (r.openmp_speedup > max_openmp_speedup)
            max_openmp_speedup = r.openmp_speedup;
        if (r.cuda_speedup > max_cuda_speedup)
            max_cuda_speedup = r.cuda_speedup;
    }

    double avg_openmp_speedup = total_openmp_speedup / results.size();
    double avg_cuda_speedup = total_cuda_speedup / results.size();

    std::cout << "\nPerformance Statistics:\n";
    std::cout << "  OpenMP Average Speedup:  " << std::fixed << std::setprecision(2)
              << avg_openmp_speedup << "x\n";
    std::cout << "  OpenMP Maximum Speedup:  " << max_openmp_speedup << "x\n";
    std::cout << "  CUDA Average Speedup:    " << avg_cuda_speedup << "x\n";
    std::cout << "  CUDA Maximum Speedup:    " << max_cuda_speedup << "x\n";
    std::cout << "======================================================================\n";

    std::string csv_filename = "benchmark_cuda_results.csv";
    save_results_csv(results, csv_filename);
    std::cout << "\nResults saved to " << csv_filename << "\n";

    return 0;
}
