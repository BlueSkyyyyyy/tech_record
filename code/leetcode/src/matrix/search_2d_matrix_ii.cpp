// 240. 搜索二维矩阵 II
// 见 search_2d_matrix_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool searchMatrix(const std::vector<std::vector<int>> &matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) {
        return false;
    }
    int i = 0;
    int j = static_cast<int>(matrix[0].size()) - 1;
    while (i < static_cast<int>(matrix.size()) && j >= 0) {
        if (matrix[i][j] == target) {
            return true;
        } else if (matrix[i][j] > target) {
            --j;
        } else {
            ++i;
        }
    }
    return false;
}

int main() {
    std::vector<std::vector<int>> matrix = {
        {1, 4, 7, 11, 15},
        {2, 5, 8, 12, 19},
        {3, 6, 9, 16, 22},
        {10, 13, 14, 17, 24},
        {18, 21, 23, 26, 30},
    };
    assert(searchMatrix(matrix, 5));
    assert(searchMatrix(matrix, 30));
    assert(searchMatrix(matrix, 1));
    assert(!searchMatrix(matrix, 20));
    assert(!searchMatrix(matrix, 0));
    assert(!searchMatrix(matrix, 31));
    assert(searchMatrix({{1}}, 1));
    assert(!searchMatrix({{1}}, 2));
    assert(!searchMatrix({}, 1));

    std::cout << "search_matrix: all tests passed\n";
    return 0;
}
