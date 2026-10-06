// 63. 不同路径 II
// 见 unique_paths_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int uniquePathsWithObstacles(const std::vector<std::vector<int>> &obstacleGrid) {
    int m = static_cast<int>(obstacleGrid.size());
    int n = static_cast<int>(obstacleGrid[0].size());
    if (obstacleGrid[0][0] == 1) return 0;
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    dp[0][0] = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (obstacleGrid[i][j] == 1) {
                dp[i][j] = 0;
                continue;
            }
            if (i > 0) dp[i][j] += dp[i - 1][j];
            if (j > 0) dp[i][j] += dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}

int main() {
    std::vector<std::vector<int>> grid1 = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    assert(uniquePathsWithObstacles(grid1) == 2);
    std::vector<std::vector<int>> grid2 = {{0, 1}, {0, 0}};
    assert(uniquePathsWithObstacles(grid2) == 1);
    std::vector<std::vector<int>> grid3 = {{0, 0}};
    assert(uniquePathsWithObstacles(grid3) == 1);
    std::vector<std::vector<int>> grid4 = {{1}};
    assert(uniquePathsWithObstacles(grid4) == 0);
    std::vector<std::vector<int>> grid5 = {{0, 0}, {1, 1}, {0, 0}};
    assert(uniquePathsWithObstacles(grid5) == 0);
    std::cout << "unique_paths_ii: all tests passed\n";
    return 0;
}
