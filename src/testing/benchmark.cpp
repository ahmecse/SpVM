#include "SparseMatrix.h"
#include "serial_spmv.h"
#include "parallel_spmv.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <omp.h>

struct BenchmarkResult {
    int size;
    int nnz;
    double density;
    double serial_time;
    double parallel_time;
    double speedup;
    double efficiency;
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

    // Measure parallel time
    double parallel_time = measure_time([&]()
                                        { spmv_parallel(matrix, vector, num_threads); });

    double speedup = serial_time / parallel_time;
    double efficiency = speedup / num_threads;

    BenchmarkResult result;
    result.size = size;
    result.nnz = matrix.getNnz();
    result.density = matrix.getDensity();
    result.serial_time = serial_time;
    result.parallel_time = parallel_time;
    result.speedup = speedup;
    result.efficiency = efficiency;
    result.num_threads = num_threads;

    return result;
}

void save_results_csv(const std::vector<BenchmarkResult> &results, const std::string &filename) {
    std::ofstream file(filename);
    file << "size,nnz,density,serial_time_ms,parallel_time_ms,speedup,efficiency,num_threads\n";
    for (const auto &r : results) {
        file << r.size << ","
             << r.nnz << ","
             << std::fixed << std::setprecision(6) << r.density << ","
             << r.serial_time * 1000 << ","
             << r.parallel_time * 1000 << ","
             << r.speedup << ","
             << r.efficiency << ","
             << r.num_threads << "\n";
    }
    file.close();
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "   SPARSE MATRIX-VECTOR MULTIPLICATION PERFORMANCE BENCHMARK\n";
    std::cout << "======================================================================\n";

    std::vector<int> sizes = {100, 200, 500, 1000, 2000, 5000, 10000};
    double density = 0.1;
    int num_threads = omp_get_max_threads();

    std::cout << "Configuration:\n";
    std::cout << "  Density: " << density * 100 << "%\n";
    std::cout << "  Number of Threads: " << num_threads << "\n";
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
        std::cout << "  Non-zeros: " << result.nnz << "\n";
        std::cout << "  Serial Time:    " << std::fixed << std::setprecision(3)
                  << result.serial_time * 1000 << " ms\n";
        std::cout << "  Parallel Time:  " << result.parallel_time * 1000 << " ms\n";
        std::cout << "  Speedup:        " << std::setprecision(2) << result.speedup << "x\n";
        std::cout << "  Efficiency:     " << std::setprecision(1)
                  << result.efficiency * 100 << "%\n\n";
    }

    std::cout << "======================================================================\n";
    std::cout << "                    BENCHMARK RESULTS SUMMARY\n";
    std::cout << "======================================================================\n";
    std::cout << std::left << std::setw(8) << "Size"
              << std::setw(12) << "NNZ"
              << std::setw(15) << "Serial(ms)"
              << std::setw(15) << "Parallel(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Efficiency" << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (const auto &r : results) {
        std::cout << std::left << std::setw(8) << r.size
                  << std::setw(12) << r.nnz
                  << std::fixed << std::setprecision(3)
                  << std::setw(15) << r.serial_time * 1000
                  << std::setw(15) << r.parallel_time * 1000
                  << std::setprecision(2)
                  << std::setw(10) << r.speedup
                  << std::setprecision(1)
                  << std::setw(12) << (r.efficiency * 100) << "%\n";
    }
    std::cout << "======================================================================\n";

    double total_speedup = 0.0;
    double max_speedup = 0.0;
    double total_efficiency = 0.0;

    for (const auto &r : results) {
        total_speedup += r.speedup;
        if (r.speedup > max_speedup)
            max_speedup = r.speedup;
        total_efficiency += r.efficiency;
    }

    double avg_speedup = total_speedup / results.size();
    double avg_efficiency = total_efficiency / results.size();
    std::cout << "\nPerformance Statistics:\n";
    std::cout << "  Average Speedup:   " << std::fixed << std::setprecision(2)
              << avg_speedup << "x\n";
    std::cout << "  Maximum Speedup:   " << max_speedup << "x\n";
    std::cout << "  Average Efficiency: " << std::setprecision(1)
              << avg_efficiency * 100 << "%\n";
    std::cout << "======================================================================\n";

    std::string csv_filename = "benchmark_results.csv";
    save_results_csv(results, csv_filename);
    std::cout << "\nResults saved to " << csv_filename << "\n";

    return 0;
}
