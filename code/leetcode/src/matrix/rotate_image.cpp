// 48. 旋转图像
// 见 rotate_image.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

void rotate(std::vector<std::vector<int>> &matrix) {
    int n = static_cast<int>(matrix.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
    for (int i = 0; i < n; ++i) {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}

int main() {
    std::vector<std::vector<int>> m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<int>> want1 = {{7, 4, 1}, {8, 5, 2}, {9, 6, 3}};
    rotate(m1);
    assert(m1 == want1);

    std::vector<std::vector<int>> m2 = {
        {5, 1, 9, 11}, {2, 4, 8, 10}, {13, 3, 6, 7}, {15, 14, 12, 16}};
    std::vector<std::vector<int>> want2 = {
        {15, 13, 2, 5}, {14, 3, 4, 1}, {12, 6, 8, 9}, {16, 7, 10, 11}};
    rotate(m2);
    assert(m2 == want2);

    std::vector<std::vector<int>> m3 = {{1}};
    std::vector<std::vector<int>> want3 = {{1}};
    rotate(m3);
    assert(m3 == want3);

    std::vector<std::vector<int>> m4 = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> want4 = {{3, 1}, {4, 2}};
    rotate(m4);
    assert(m4 == want4);

    std::cout << "rotate: all tests passed\n";
    return 0;
}
