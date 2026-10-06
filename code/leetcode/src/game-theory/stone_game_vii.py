"""1690. 石子游戏 VII（Stone Game VII）

题目：若干堆石子排成一行 stones。Alice 先手，两人轮流从**当前两端**拿走一堆，这一手
的得分是「拿完之后，剩下的所有石子数量之和」（也就是说，拿走的这堆不计入得分，反而让
后面的人少拿）。两人最优策略，问 Alice 相对 Bob 的**最大分差**。

思路（区间 DP，得分是「剩下的和」）：
    和 486 一样只记分差，但「得分」的定义变了：取走一堆后，得分为剩余石子之和。设
    dp[i][j] 表示在子数组 stones[i..j] 上行动者能领先的最大分差。若取左端，本次得分是
    sum(i+1..j)，随后对手在 [i+1, j] 上能领先 dp[i+1][j]，净分差为
    sum(i+1..j) - dp[i+1][j]；取右端同理：

        dp[i][j] = max( sum(i+1..j) - dp[i+1][j], sum(i..j-1) - dp[i][j-1] )

    区间长度为 1 时，拿走这唯一的堆后剩余为 0，得分 0，故 dp[i][i] = 0。区间和用前缀和
    O(1) 求出。最终答案是 dp[0][n-1]。

    为什么这里 base 是 0 而不是元素值：因为「拿走最后一堆」时已经没有剩余石子，本次得分
    为 0，这一手不带来任何分差。这正体现了本题得分规则与 486 / 877 的区别。

复杂度：时间 O(n^2)，空间 O(n^2)（可用一维滚动优化到 O(n)）。
"""


def stone_game_vii(stones):
    n = len(stones)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]

    def range_sum(i, j):
        return prefix[j + 1] - prefix[i]

    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            left = range_sum(i + 1, j) - dp[i + 1][j]
            right = range_sum(i, j - 1) - dp[i][j - 1]
            dp[i][j] = max(left, right)
    return dp[0][n - 1]


if __name__ == "__main__":
    assert stone_game_vii([5, 3, 1, 4, 2]) == 6
    assert stone_game_vii([7, 90, 5, 1, 100, 10, 10, 0]) == 122
    assert stone_game_vii([1, 1]) == 1
    assert stone_game_vii([1, 2, 3]) == 2
    print("stone_game_vii: all tests passed")
