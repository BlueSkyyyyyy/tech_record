"""63. 不同路径 II（Unique Paths II）

题目：在 m x n 网格中，1 表示障碍物、0 表示空地。机器人从左上角出发，
每次只能向下或向右，求到达右下角的路径数（路径上不能有障碍物）。

思路（在 62 的网格 DP 上「挖洞」）：
    转移和 62 完全一样：dp[i][j] = 上方路径数 + 左方路径数。
    区别只有一处：若 (i, j) 是障碍物，则 dp[i][j] = 0——
    因为没有任何路径能停在障碍物上，所有经过它的方案自然断掉。

    直接按「先看是不是障碍物，再加两个方向」来写，就不必单独处理
    第一行/第一列被障碍物截断的情况：障碍物之后的格子会因为前驱为 0
    而自动变成 0。

    起始格若本身是障碍物，直接返回 0。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def unique_paths_with_obstacles(obstacle_grid):
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    if obstacle_grid[0][0] == 1:
        return 0
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                dp[i][j] = 0
                continue
            if i > 0:
                dp[i][j] += dp[i - 1][j]
            if j > 0:
                dp[i][j] += dp[i][j - 1]
    return dp[m - 1][n - 1]


if __name__ == "__main__":
    grid1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert unique_paths_with_obstacles(grid1) == 2
    grid2 = [[0, 1], [0, 0]]
    assert unique_paths_with_obstacles(grid2) == 1
    grid3 = [[0, 0]]
    assert unique_paths_with_obstacles(grid3) == 1
    grid4 = [[1]]
    assert unique_paths_with_obstacles(grid4) == 0
    grid5 = [[0, 0], [1, 1], [0, 0]]
    assert unique_paths_with_obstacles(grid5) == 0
    print("unique_paths_ii: all tests passed")
