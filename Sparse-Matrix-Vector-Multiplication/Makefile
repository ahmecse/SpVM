# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -Isrc -Isrc/serial -Isrc/openmp
LDFLAGS = -fopenmp

# Source directories
SRC_DIR = src
SERIAL_DIR = $(SRC_DIR)/serial
OPENMP_DIR = $(SRC_DIR)/openmp
TESTING_DIR = $(SRC_DIR)/testing
PLOTTING_DIR = $(SRC_DIR)/plotting

# Source files
SPARSE_MATRIX_SRC = $(SRC_DIR)/SparseMatrix.cpp
SERIAL_SRC = $(SERIAL_DIR)/serial_spmv.cpp
OPENMP_SRC = $(OPENMP_DIR)/parallel_spmv.cpp

SOURCES = $(SPARSE_MATRIX_SRC) $(SERIAL_SRC) $(OPENMP_SRC)
OBJECTS = SparseMatrix.o serial_spmv.o parallel_spmv.o

# Headers
SPARSE_MATRIX_HDR = $(SRC_DIR)/SparseMatrix.h
SERIAL_HDR = $(SERIAL_DIR)/serial_spmv.h
OPENMP_HDR = $(OPENMP_DIR)/parallel_spmv.h
HEADERS = $(SPARSE_MATRIX_HDR) $(SERIAL_HDR) $(OPENMP_HDR)

# Executables
TARGETS = validate benchmark

# Default target
all: $(TARGETS)

# Validate program
validate: validate.o $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Benchmark program
benchmark: benchmark.o $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Object file compilation
SparseMatrix.o: $(SPARSE_MATRIX_SRC) $(SPARSE_MATRIX_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

serial_spmv.o: $(SERIAL_SRC) $(SERIAL_HDR) $(SPARSE_MATRIX_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

parallel_spmv.o: $(OPENMP_SRC) $(OPENMP_HDR) $(SPARSE_MATRIX_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

validate.o: $(TESTING_DIR)/validate.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

benchmark.o: $(TESTING_DIR)/benchmark.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) validate.o benchmark.o $(TARGETS)
	rm -f benchmark_results.csv
	rm -rf results/

# Clean and rebuild
rebuild: clean all

# Run validation tests
test: validate
	./validate

# Run benchmarks
bench: benchmark
	./benchmark

# Generate plots (requires Python with matplotlib and pandas)
plot: bench
	python3 $(PLOTTING_DIR)/plot_results.py

# Full workflow: build, test, benchmark, and plot
run: all test bench plot

.PHONY: all clean rebuild test bench plot run
