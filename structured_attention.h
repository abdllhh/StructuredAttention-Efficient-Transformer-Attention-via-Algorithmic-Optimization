#pragma once

#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

namespace sa { 

class Matrix {
public:
    // Const
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float value);
    
    // Basic operations
    float at(size_t row, size_t col) const;
    void set(size_t row, size_t col, float value);
    std::vector<float> getRow(size_t row) const;
    
    // Matrix operations
    Matrix transpose() const;
    Matrix matmul(const Matrix& other) const;
    Matrix elementwiseMul(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix scale(float scalar) const;
    Matrix softmax() const;
    
    // Static factory methods
    static Matrix random(size_t rows, size_t cols, float min = -1.0f, float max = 1.0f);
    
    // Dimension information
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
};

// ============= KD-Tree Class =============
struct KDNode {
    std::vector<float> point;
    int idx;
    KDNode* left;
    KDNode* right;
    
    KDNode(const std::vector<float>& p, int index);
    ~KDNode();
};

class KDTree {
public:
    KDTree(const std::vector<std::vector<float>>& points);
    ~KDTree();
    std::vector<int> findKNearest(const std::vector<float>& query, int k) const;
    
private:
    KDNode* root_;
    std::vector<std::vector<float>> points_;
    
    KDNode* buildTree(const std::vector<int>& indices, int depth);
    void searchKNN(const KDNode* node, const std::vector<float>& query, int k,
                  std::priority_queue<std::pair<float, int>>& pq, int depth) const;
    float distanceSquared(const std::vector<float>& a, const std::vector<float>& b) const;
};

// ============= Base Attention Class =============
class BaseAttention {
public:
    BaseAttention() = default;
    virtual ~BaseAttention() = default;
    
    // Compute attention
    virtual Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V);
    
    // Helper methods
    Matrix calculateScores(const Matrix& Q, const Matrix& K);
    Matrix calculateWeights(const Matrix& scores);
};

// ============= KD-Tree Optimized Attention Class =============
class KDAttention : public BaseAttention {
public:
    KDAttention(int topK = 16);
    ~KDAttention() = default;
    
    // Optimized attention using KD-Tree
    Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V) override;
    
private:
    int topK_;
};

// Helper functions for printing
void printMatrix(const Matrix& mat, const std::string& name);

} // namespace sa