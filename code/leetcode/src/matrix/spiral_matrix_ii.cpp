// 59. 螺旋矩阵 II
// 见 spiral_matrix_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> generateMatrix(int n) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    int top = 0, bottom = n - 1, left = 0, right = n - 1;
    int num = 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) {
            matrix[top][j] = num++;
        }
        ++top;
        for (int i = top; i <= bottom; ++i) {
            matrix[i][right] = num++;
        }
        --right;
        if (top <= bottom) {
            for (int j = right; j >= left; --j) {
                matrix[bottom][j] = num++;
            }
            --bottom;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; --i) {
                matrix[i][left] = num++;
            }
            ++left;
        }
    }
    return matrix;
}

int main() {
    std::vector<std::vector<int>> want1 = {{1}};
    assert(generateMatrix(1) == want1);

    std::vector<std::vector<int>> want2 = {{1, 2}, {4, 3}};
    assert(generateMatrix(2) == want2);

    std::vector<std::vector<int>> want3 = {{1, 2, 3}, {8, 9, 4}, {7, 6, 5}};
    assert(generateMatrix(3) == want3);

    std::vector<std::vector<int>> want4 = {
        {1, 2, 3, 4}, {12, 13, 14, 5}, {11, 16, 15, 6}, {10, 9, 8, 7}};
    assert(generateMatrix(4) == want4);

    std::cout << "generate_matrix: all tests passed\n";
    return 0;
}
