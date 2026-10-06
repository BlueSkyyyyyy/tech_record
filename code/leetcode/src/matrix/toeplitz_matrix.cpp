// 766. 托普利茨矩阵
// 见 toeplitz_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool isToeplitzMatrix(const std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][j] != matrix[i - 1][j - 1]) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    assert(isToeplitzMatrix({{1, 2, 3, 4}, {5, 1, 2, 3}, {9, 5, 1, 2}}));
    assert(!isToeplitzMatrix({{1, 2}, {2, 2}}));
    assert(isToeplitzMatrix({{1}}));
    assert(isToeplitzMatrix({{1, 2}, {3, 1}}));
    assert(!isToeplitzMatrix({{1, 2, 3}, {4, 5, 1}}));

    std::cout << "is_toeplitz_matrix: all tests passed\n";
    return 0;
}
