#include "structured_attention.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>

using namespace sa;

struct BenchmarkResult {
    size_t seq_len;
    size_t dim;
    double std_time_ms;
    double kd_time_ms;
    double speedup;
    double error;
};

std::vector<BenchmarkResult> results;

void runBenchmark(size_t seq_len, size_t dim) {
    std::cout << "=======================================" << std::endl;
    std::cout << "Benchmark: seq_len=" << seq_len << ", dim=" << dim << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Create random matrices
    Matrix Q = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    Matrix K = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    Matrix V = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    
    // Standard attention
    BaseAttention stdAttn;
    auto start = std::chrono::high_resolution_clock::now();
    Matrix stdOutput = stdAttn.forward(Q, K, V);
    auto end = std::chrono::high_resolution_clock::now();
    auto stdTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // KD-Tree optimized attention
    KDAttention kdAttn(16); // Use top-16 neighbors
    start = std::chrono::high_resolution_clock::now();
    Matrix kdOutput = kdAttn.forward(Q, K, V);
    end = std::chrono::high_resolution_clock::now();
    auto kdTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Calculate error
    float error = 0.0f;
    for (size_t i = 0; i < stdOutput.rows(); ++i) {
        for (size_t j = 0; j < stdOutput.cols(); ++j) {
            float diff = stdOutput.at(i, j) - kdOutput.at(i, j);
            error += diff * diff;
        }
    }
    error = std::sqrt(error) / (stdOutput.rows() * stdOutput.cols());
    
    // Print results
    std::cout << "Standard Attention: " << stdTime << " ms" << std::endl;
    std::cout << "KD-Tree Attention: " << kdTime << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<float>(stdTime) / kdTime << "x" << std::endl;
    std::cout << "Approximation Error: " << error << std::endl;
    std::cout << std::endl;
    
    // Store results
    BenchmarkResult result;
    result.seq_len = seq_len;
    result.dim = dim;
    result.std_time_ms = stdTime;
    result.kd_time_ms = kdTime;
    result.speedup = static_cast<double>(stdTime) / kdTime;
    result.error = error;
    results.push_back(result);
}

void saveResultsToCSV(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "sequence_length,dimension,standard_attention_ms,kd_tree_attention_ms,speedup,approximation_error" << std::endl;
    
    // Write data
    for (const auto& result : results) {
        file << result.seq_len << ","
             << result.dim << ","
             << result.std_time_ms << ","
             << result.kd_time_ms << ","
             << result.speedup << ","
             << result.error << std::endl;
    }
    
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {
    std::cout << "StructuredAttention: Efficient Transformer Attention via Algorithmic Optimization" << std::endl;
    std::cout << "================================================================================" << std::endl;
    
    // Run simple test first
    size_t seq_len = 4;
    size_t dim = 8;
    
    // Create test matrices
    Matrix Q = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    Matrix K = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    Matrix V = Matrix::random(seq_len, dim, -1.0f, 1.0f);
    
    std::cout << "Test matrices created:" << std::endl;
    printMatrix(Q, "Q");
    printMatrix(K, "K");
    printMatrix(V, "V");
    
    // Test standard attention
    std::cout << "Computing standard attention..." << std::endl;
    BaseAttention baseAttn;
    Matrix baseOutput = baseAttn.forward(Q, K, V);
    printMatrix(baseOutput, "Standard Attention Output");
    
    // Test KD-Tree optimized attention
    std::cout << "Computing KD-Tree optimized attention..." << std::endl;
    KDAttention kdAttn(2); // Use top-2 neighbors for small test
    Matrix kdOutput = kdAttn.forward(Q, K, V);
    printMatrix(kdOutput, "KD-Tree Attention Output");
    
    // Run benchmarks with more sequence lengths
    std::cout << "\nRunning benchmarks...\n" << std::endl;
    
    // Fixed dimension with varying sequence lengths
    const size_t fixed_dim = 64;
    std::vector<size_t> seq_lengths = {16, 32, 64, 96, 128, 160, 192, 224, 256};
    
    for (const auto& len : seq_lengths) {
        runBenchmark(len, fixed_dim);
    }
    
    // Save results to CSV file
    saveResultsToCSV("attention_benchmark_results.csv");
    
    return 0;
}