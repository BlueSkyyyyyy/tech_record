// 221. 最大正方形
// 见 maximal_square.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int maximalSquare(const std::vector<std::vector<char>> &matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (matrix[i - 1][j - 1] == '1') {
                dp[i][j] = std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                best = std::max(best, dp[i][j]);
            }
        }
    }
    return best * best;
}

int main() {
    std::vector<std::vector<char>> m1 = {
        {'1', '0', '1', '0', '0'},
        {'1', '0', '1', '1', '1'},
        {'1', '1', '1', '1', '1'},
        {'1', '0', '0', '1', '0'},
    };
    assert(maximalSquare(m1) == 4);
    std::vector<std::vector<char>> m2 = {{'0', '1'}, {'1', '0'}};
    assert(maximalSquare(m2) == 1);
    std::vector<std::vector<char>> m3 = {{'0'}};
    assert(maximalSquare(m3) == 0);
    std::vector<std::vector<char>> m4 = {{'1', '1'}, {'1', '1'}};
    assert(maximalSquare(m4) == 4);
    std::cout << "maximal_square: all tests passed\n";
    return 0;
}
