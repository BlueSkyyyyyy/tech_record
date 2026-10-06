// 498. 对角线遍历
// 见 diagonal_traverse.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<int> findDiagonalOrder(const std::vector<std::vector<int>> &mat) {
    int m = static_cast<int>(mat.size());
    int n = static_cast<int>(mat[0].size());
    std::vector<int> res;
    for (int d = 0; d < m + n - 1; ++d) {
        if (d % 2 == 0) {
            int r = std::min(d, m - 1);
            int c = d - r;
            while (r >= 0 && c < n) {
                res.push_back(mat[r][c]);
                --r;
                ++c;
            }
        } else {
            int c = std::min(d, n - 1);
            int r = d - c;
            while (c >= 0 && r < m) {
                res.push_back(mat[r][c]);
                ++r;
                --c;
            }
        }
    }
    return res;
}

int main() {
    std::vector<int> want1 = {1, 2, 4, 7, 5, 3, 6, 8, 9};
    assert(findDiagonalOrder({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}) == want1);

    std::vector<int> want2 = {1, 2, 3, 4};
    assert(findDiagonalOrder({{1, 2}, {3, 4}}) == want2);

    std::vector<int> want3 = {1, 2, 3};
    assert(findDiagonalOrder({{1, 2, 3}}) == want3);

    std::vector<int> want4 = {1, 2, 3};
    assert(findDiagonalOrder({{1}, {2}, {3}}) == want4);

    std::vector<int> want5 = {1, 2, 5, 6, 3, 4, 7, 8};
    assert(findDiagonalOrder({{1, 2, 3, 4}, {5, 6, 7, 8}}) == want5);

    std::cout << "find_diagonal_order: all tests passed\n";
    return 0;
}
