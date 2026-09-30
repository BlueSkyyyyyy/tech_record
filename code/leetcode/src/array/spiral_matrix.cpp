// 54. 螺旋矩阵
// 见 spiral_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<int> spiralOrder(const std::vector<std::vector<int>> &matrix) {
    std::vector<int> res;
    if (matrix.empty() || matrix[0].empty()) return res;
    int top = 0, bottom = static_cast<int>(matrix.size()) - 1;
    int left = 0, right = static_cast<int>(matrix[0].size()) - 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) res.push_back(matrix[top][j]);
        ++top;
        for (int i = top; i <= bottom; ++i) res.push_back(matrix[i][right]);
        --right;
        if (top <= bottom) {
            for (int j = right; j >= left; --j) res.push_back(matrix[bottom][j]);
            --bottom;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; --i) res.push_back(matrix[i][left]);
            ++left;
        }
    }
    return res;
}

int main() {
    {
        std::vector<std::vector<int>> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::vector<int> want{1, 2, 3, 6, 9, 8, 7, 4, 5};
        assert(spiralOrder(m) == want);
    }
    {
        std::vector<std::vector<int>> m{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
        std::vector<int> want{1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7};
        assert(spiralOrder(m) == want);
    }
    {
        std::vector<std::vector<int>> m{{1}};
        std::vector<int> want{1};
        assert(spiralOrder(m) == want);
    }
    {
        std::vector<std::vector<int>> m{{1, 2}, {3, 4}};
        std::vector<int> want{1, 2, 4, 3};
        assert(spiralOrder(m) == want);
    }
    std::vector<std::vector<int>> empty;
    assert(spiralOrder(empty).empty());
    {
        std::vector<std::vector<int>> m{{}};
        assert(spiralOrder(m).empty());
    }
    std::cout << "spiral_matrix: all tests passed\n";
    return 0;
}
