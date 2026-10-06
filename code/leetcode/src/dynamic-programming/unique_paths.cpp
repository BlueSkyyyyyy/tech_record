// 62. 不同路径
// 见 unique_paths.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int uniquePaths(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 1));
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}

int main() {
    assert(uniquePaths(3, 7) == 28);
    assert(uniquePaths(3, 2) == 3);
    assert(uniquePaths(1, 1) == 1);
    assert(uniquePaths(1, 5) == 1);
    assert(uniquePaths(5, 1) == 1);
    std::cout << "unique_paths: all tests passed\n";
    return 0;
}
