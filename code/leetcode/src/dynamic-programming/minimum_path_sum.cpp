// 64. 最小路径和
// 见 minimum_path_sum.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int minPathSum(const std::vector<std::vector<int>> &grid) {
    int m = static_cast<int>(grid.size());
    int n = static_cast<int>(grid[0].size());
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; ++i) dp[i][0] = dp[i - 1][0] + grid[i][0];
    for (int j = 1; j < n; ++j) dp[0][j] = dp[0][j - 1] + grid[0][j];
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = grid[i][j] + std::min(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[m - 1][n - 1];
}

int main() {
    std::vector<std::vector<int>> grid1 = {{1, 3, 1}, {1, 5, 1}, {4, 2, 1}};
    assert(minPathSum(grid1) == 7);
    std::vector<std::vector<int>> grid2 = {{1, 2, 3}, {4, 5, 6}};
    assert(minPathSum(grid2) == 12);
    std::vector<std::vector<int>> grid3 = {{5}};
    assert(minPathSum(grid3) == 5);
    std::cout << "minimum_path_sum: all tests passed\n";
    return 0;
}
