// 73. 矩阵置零
// 见 set_matrix_zeroes.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void setZeroes(std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());

    bool col0 = false;
    for (int i = 0; i < m; ++i) {
        if (matrix[i][0] == 0) {
            col0 = true;
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                matrix[i][j] = 0;
            }
        }
    }
    if (matrix[0][0] == 0) {
        for (int j = 0; j < n; ++j) {
            matrix[0][j] = 0;
        }
    }
    if (col0) {
        for (int i = 0; i < m; ++i) {
            matrix[i][0] = 0;
        }
    }
}

int main() {
    std::vector<std::vector<int>> m1 = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    std::vector<std::vector<int>> want1 = {{1, 0, 1}, {0, 0, 0}, {1, 0, 1}};
    setZeroes(m1);
    assert(m1 == want1);

    std::vector<std::vector<int>> m2 = {{0, 1, 2, 0}, {3, 4, 5, 2}, {1, 3, 1, 5}};
    std::vector<std::vector<int>> want2 = {{0, 0, 0, 0}, {0, 4, 5, 0}, {0, 3, 1, 0}};
    setZeroes(m2);
    assert(m2 == want2);

    std::vector<std::vector<int>> m3 = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> want3 = {{1, 2, 3}, {4, 5, 6}};
    setZeroes(m3);
    assert(m3 == want3);

    std::vector<std::vector<int>> m4 = {{1}, {0}};
    std::vector<std::vector<int>> want4 = {{0}, {0}};
    setZeroes(m4);
    assert(m4 == want4);

    std::cout << "set_zeroes: all tests passed\n";
    return 0;
}
