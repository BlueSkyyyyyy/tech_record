"""64. 最小路径和（Minimum Path Sum）

题目：给一个 m x n 的非负整数网格 grid，从左上角走到右下角，每次只能向下
或向右，求路径上的数字总和的最小值。

思路（网格 DP：求最小用 min）：
    dp[i][j] = 从左上角走到 (i, j) 的最小路径和。
    到达 (i, j) 的最后一步只能来自上方或左方，代价是 grid[i][j]，故

        dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    边界：第一行只能从左边来、第一列只能从上面来，各自累加即可。

    「数方案」用加法（62），「求最小代价」用 min——网格没变，
    变的只是聚合方式和「是否加上当前格的权重」。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[m - 1][n - 1]


if __name__ == "__main__":
    grid1 = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
    assert min_path_sum(grid1) == 7
    grid2 = [[1, 2, 3], [4, 5, 6]]
    assert min_path_sum(grid2) == 12
    grid3 = [[5]]
    assert min_path_sum(grid3) == 5
    print("minimum_path_sum: all tests passed")
