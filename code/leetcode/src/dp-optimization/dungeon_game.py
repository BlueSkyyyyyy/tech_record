"""174. 地下城游戏（Dungeon Game）

题目：骑士从左上角走到右下角（只能向右或向下），每个格子会加血（正）或掉血（负）。
任意时刻血量必须 >= 1。求到达终点所需的最小初始血量。

思路（从终点倒推 + 空间压缩）：
    正向不好做，因为「当前血量」是过程量、会污染状态。改为倒推：
    设 dp[i][j] = 从 (i,j) 走到终点所需的最小初始血量。则
        dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
    含义：先保证到下一步的存活需求取「更省的那条路」，减去本格增减后，
    至少为 1。
    由于 dp[i][j] 只依赖右、下，可以用**一维滚动数组**把空间从 O(m·n) 压到 O(n)：
    每行从右向左更新，dp[j] 更新前代表「下一行同列」，dp[j+1] 代表「本行右一列」。

复杂度：时间 O(m·n)，空间 O(n)。
"""


def calculate_minimum_hp(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[0] * n for _ in range(m)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                need = 1 - dungeon[i][j]
            elif i == m - 1:
                need = dp[i][j + 1] - dungeon[i][j]
            elif j == n - 1:
                need = dp[i + 1][j] - dungeon[i][j]
            else:
                need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
            dp[i][j] = max(1, need)
    return dp[0][0]


def calculate_minimum_hp_1d(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[n - 1] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[j] = max(1, min(dp[j], dp[j + 1]) - dungeon[i][j])
        dp[n] = INF
    return dp[0]


if __name__ == "__main__":
    d = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
    assert calculate_minimum_hp(d) == 7
    assert calculate_minimum_hp_1d(d) == 7
    assert calculate_minimum_hp([[0]]) == 1
    assert calculate_minimum_hp([[-3, 5]]) == 4
    assert calculate_minimum_hp_1d([[100]]) == 1
    print("dungeon_game: all tests passed")
