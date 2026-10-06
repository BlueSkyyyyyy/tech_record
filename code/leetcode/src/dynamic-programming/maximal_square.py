"""221. 最大正方形（Maximal Square）

题目：在一个由 '0' 和 '1' 组成的二维矩阵中，找出只包含 '1' 的最大正方形，
返回它的面积。

思路（以「右下角」为状态的网格 DP）：
    dp[i][j] = 以 (i, j) 为右下角、且全部为 '1' 的最大正方形的边长。
    若 grid[i][j] == '1'，它能向右上、左上、正上三个方向「借」边长：

        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    取三者最小：一块正方形要同时被上、左、左上三块覆盖，短板决定边长。
    若当前格是 '0'，dp[i][j] = 0。过程中记录最大边长，答案是其平方。

    用滚动的一维数组时，需要先备份「左上角」的旧值，因为 dp[j-1] 会被本行覆盖。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == "1":
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                best = max(best, dp[i][j])
    return best * best


if __name__ == "__main__":
    m1 = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "1", "0"],
    ]
    assert maximal_square(m1) == 4
    m2 = [["0", "1"], ["1", "0"]]
    assert maximal_square(m2) == 1
    m3 = [["0"]]
    assert maximal_square(m3) == 0
    m4 = [["1", "1"], ["1", "1"]]
    assert maximal_square(m4) == 4
    print("maximal_square: all tests passed")
