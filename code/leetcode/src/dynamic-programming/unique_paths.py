"""62. 不同路径（Unique Paths）

题目：一个机器人位于 m x n 网格的左上角，每次只能向下或向右移动一步，
问到达右下角共有多少条不同的路径。

思路（二维网格 DP：数方案用加法）：
    dp[i][j] = 从左上角走到第 i 行第 j 列的路径数。
    机器人只能从上方 (i-1, j) 或左方 (i, j-1) 进入，故

        dp[i][j] = dp[i-1][j] + dp[i][j-1]

    边界：第一行和第一列都只有唯一一条直线路径，全部为 1。

为什么第一行/第一列是 1：站在第一行的任意格子，只能一路向右走过来，
只有这一种走法；第一列同理。它们不是「没有前驱」，而是「前驱唯一」。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]


if __name__ == "__main__":
    assert unique_paths(3, 7) == 28
    assert unique_paths(3, 2) == 3
    assert unique_paths(1, 1) == 1
    assert unique_paths(1, 5) == 1
    assert unique_paths(5, 1) == 1
    print("unique_paths: all tests passed")
