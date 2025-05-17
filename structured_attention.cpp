#include "structured_attention.h"

namespace sa {

// =============== Matrix Implementation ===============
Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Matrix::Matrix(size_t rows, size_t cols, float value) 
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

float Matrix::at(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data_[row * cols_ + col];
}

void Matrix::set(size_t row, size_t col, float value) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    data_[row * cols_ + col] = value;
}

std::vector<float> Matrix::getRow(size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    std::vector<float> result(cols_);
    for (size_t i = 0; i < cols_; ++i) {
        result[i] = at(row, i);
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.set(j, i, at(i, j));
        }
    }
    return result;
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols_ != other.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    Matrix result(rows_, other.cols());
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

Matrix Matrix::elementwiseMul(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for elementwise multiplication");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.set(i, j, at(i, j) * other.at(i, j));
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for addition");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.set(i, j, at(i, j) + other.at(i, j));
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows_ != other.rows() || cols_ != other.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for subtraction");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.set(i, j, at(i, j) - other.at(i, j));
        }
    }
    return result;
}

Matrix Matrix::scale(float scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.set(i, j, at(i, j) * scalar);
        }
    }
    return result;
}

Matrix Matrix::softmax() const {
    Matrix result(rows_, cols_);
    
    // Apply softmax row-wise
    for (size_t i = 0; i < rows_; ++i) {
        // Find max value in this row for numerical stability
        float max_val = at(i, 0);
        for (size_t j = 1; j < cols_; ++j) {
            max_val = std::max(max_val, at(i, j));
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (size_t j = 0; j < cols_; ++j) {
            float exp_val = std::exp(at(i, j) - max_val);
            result.set(i, j, exp_val);
            sum += exp_val;
        }
        
        // Normalize
        for (size_t j = 0; j < cols_; ++j) {
            result.set(i, j, result.at(i, j) / sum);
        }
    }
    
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, float min, float max) {
    Matrix result(rows, cols);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.set(i, j, dist(gen));
        }
    }
    
    return result;
}

// =============== KDNode Implementation ===============
KDNode::KDNode(const std::vector<float>& p, int index) 
    : point(p), idx(index), left(nullptr), right(nullptr) {}

KDNode::~KDNode() {
    delete left;
    delete right;
}

// =============== KDTree Implementation ===============
KDTree::KDTree(const std::vector<std::vector<float>>& points) : root_(nullptr), points_(points) {
    if (points.empty()) return;
    
    std::vector<int> indices(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }
    
    root_ = buildTree(indices, 0);
}

KDTree::~KDTree() {
    delete root_;
}

KDNode* KDTree::buildTree(const std::vector<int>& indices, int depth) {
    if (indices.empty()) return nullptr;
    
    int dim = depth % points_[0].size();
    
    // Sort points based on the current dimension
    std::vector<int> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [this, dim](int a, int b) {
                  return points_[a][dim] < points_[b][dim];
              });
    
    // Select median as pivot
    int median_idx = sorted_indices.size() / 2;
    
    // Create node for the median point
    KDNode* node = new KDNode(points_[sorted_indices[median_idx]], sorted_indices[median_idx]);
    
    // Recursively build left and right subtrees
    std::vector<int> left_indices(sorted_indices.begin(), sorted_indices.begin() + median_idx);
    std::vector<int> right_indices(sorted_indices.begin() + median_idx + 1, sorted_indices.end());
    
    node->left = buildTree(left_indices, depth + 1);
    node->right = buildTree(right_indices, depth + 1);
    
    return node;
}

std::vector<int> KDTree::findKNearest(const std::vector<float>& query, int k) const {
    std::priority_queue<std::pair<float, int>> pq;
    searchKNN(root_, query, k, pq, 0);
    
    std::vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }
    
    return result;
}

void KDTree::searchKNN(const KDNode* node, const std::vector<float>& query, int k,
                      std::priority_queue<std::pair<float, int>>& pq, int depth) const {
    if (!node) return;
    
    // Calculate distance to current point
    float dist = distanceSquared(query, node->point);
    
    // Add current point to priority queue if:
    // 1. Queue is not full yet, or
    // 2. Current point is closer than the furthest point in the queue
    if (pq.size() < static_cast<size_t>(k) || dist < pq.top().first) {
        if (pq.size() == static_cast<size_t>(k)) pq.pop();
        pq.push({dist, node->idx});
    }
    
    // Dimension to split on
    int dim = depth % query.size();
    
    // Determine which side to search first
    KDNode* first = (query[dim] < node->point[dim]) ? node->left : node->right;
    KDNode* second = (query[dim] < node->point[dim]) ? node->right : node->left;
    
    // Search the near side first
    searchKNN(first, query, k, pq, depth + 1);
    
    // Only search the far side if necessary
    float dist_to_plane = std::abs(query[dim] - node->point[dim]);
    if (pq.size() < static_cast<size_t>(k) || dist_to_plane * dist_to_plane < pq.top().first) {
        searchKNN(second, query, k, pq, depth + 1);
    }
}

float KDTree::distanceSquared(const std::vector<float>& a, const std::vector<float>& b) const {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// =============== BaseAttention Implementation ===============
Matrix BaseAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V) {
    // Compute attention scores
    Matrix scores = calculateScores(Q, K);
    
    // Apply softmax to get attention weights
    Matrix weights = calculateWeights(scores);
    
    // Compute output
    return weights.matmul(V);
}

Matrix BaseAttention::calculateScores(const Matrix& Q, const Matrix& K) {
    // Compute Q * K^T
    Matrix scores = Q.matmul(K.transpose());
    
    // Scale by sqrt(d_k)
    float scale = 1.0f / std::sqrt(static_cast<float>(K.cols()));
    return scores.scale(scale);
}

Matrix BaseAttention::calculateWeights(const Matrix& scores) {
    // Apply softmax
    return scores.softmax();
}

// =============== KDAttention Implementation ===============
KDAttention::KDAttention(int topK) : topK_(topK) {}

Matrix KDAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V) {
    size_t batch_size = 1; // Simplified - assuming single batch
    size_t seq_len = Q.rows();
    size_t d_k = Q.cols();
    
    Matrix result(seq_len, V.cols());
    
    // Convert keys to vectors for KD-Tree
    std::vector<std::vector<float>> key_vectors;
    for (size_t i = 0; i < seq_len; ++i) {
        key_vectors.push_back(K.getRow(i));
    }
    
    // Build KD-Tree with keys
    KDTree kdtree(key_vectors);
    
    // For each query
    for (size_t i = 0; i < seq_len; ++i) {
        std::vector<float> query = Q.getRow(i);
        
        // Find top-k similar keys
        std::vector<int> nearest = kdtree.findKNearest(query, topK_);
        
        // Compute sparse attention
        std::vector<float> weights(nearest.size());
        float sum = 0.0f;
        
        // Calculate attention weights for nearest neighbors
        for (size_t j = 0; j < nearest.size(); ++j) {
            int idx = nearest[j];
            float dot_product = 0.0f;
            for (size_t d = 0; d < d_k; ++d) {
                dot_product += query[d] * K.at(idx, d);
            }
            weights[j] = std::exp(dot_product / std::sqrt(static_cast<float>(d_k)));
            sum += weights[j];
        }
        
        // Normalize weights
        for (float& w : weights) {
            w /= sum;
        }
        
        // Compute weighted sum
        for (size_t d = 0; d < V.cols(); ++d) {
            float val = 0.0f;
            for (size_t j = 0; j < nearest.size(); ++j) {
                int idx = nearest[j];
                val += weights[j] * V.at(idx, d);
            }
            result.set(i, d, val);
        }
    }
    
    return result;
}

// =============== Helper Functions ===============
void printMatrix(const Matrix& mat, const std::string& name) {
    std::cout << "Matrix " << name << " (" << mat.rows() << "x" << mat.cols() << "):" << std::endl;
    
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            std::cout << std::fixed << std::setprecision(4) << mat.at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

} // namespace sa