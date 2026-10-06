// 566. 重塑矩阵
// 见 reshape_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> matrixReshape(const std::vector<std::vector<int>> &mat,
                                            int r, int c) {
    int m = static_cast<int>(mat.size());
    int n = static_cast<int>(mat[0].size());
    if (m * n != r * c) {
        return mat;
    }
    std::vector<std::vector<int>> res(r, std::vector<int>(c, 0));
    for (int k = 0; k < m * n; ++k) {
        res[k / c][k % c] = mat[k / n][k % n];
    }
    return res;
}

int main() {
    std::vector<std::vector<int>> want1 = {{1, 2, 3, 4}};
    assert(matrixReshape({{1, 2}, {3, 4}}, 1, 4) == want1);

    std::vector<std::vector<int>> want2 = {{1, 2}, {3, 4}};
    assert(matrixReshape({{1, 2}, {3, 4}}, 2, 4) == want2);

    std::vector<std::vector<int>> want3 = {{1}, {2}, {3}, {4}};
    assert(matrixReshape({{1, 2}, {3, 4}}, 4, 1) == want3);

    std::vector<std::vector<int>> want4 = {{1, 2}, {3, 4}};
    assert(matrixReshape({{1, 2, 3, 4}}, 2, 2) == want4);

    std::cout << "matrix_reshape: all tests passed\n";
    return 0;
}
