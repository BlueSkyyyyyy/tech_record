// 867. 转置矩阵
// 见 transpose.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    std::vector<std::vector<int>> result(n, std::vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

int main() {
    std::vector<std::vector<int>> want1 = {{1, 4}, {2, 5}, {3, 6}};
    assert(transpose({{1, 2, 3}, {4, 5, 6}}) == want1);

    std::vector<std::vector<int>> want2 = {{1, 3, 5}, {2, 4, 6}};
    assert(transpose({{1, 2}, {3, 4}, {5, 6}}) == want2);

    std::vector<std::vector<int>> want3 = {{7}};
    assert(transpose({{7}}) == want3);

    std::vector<std::vector<int>> want4 = {{1}, {2}};
    assert(transpose({{1, 2}}) == want4);

    std::vector<std::vector<int>> want5 = {{1, 2}};
    assert(transpose({{1}, {2}}) == want5);

    std::cout << "transpose: all tests passed\n";
    return 0;
}
