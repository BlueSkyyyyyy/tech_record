"""877. 石子游戏（Stone Game）

题目：偶数堆石子排成一行，piles[i] 是第 i 堆的数量。两人轮流从**当前两端**拿走一整堆
并计入自己的总分，两堆都被拿完时结束，总分多者获胜。两人都最优策略，问先手（Alice）
是否必胜。

思路（与 486 同款的区间 DP，问法不同）：
    依旧是零和博弈，只记「当前行动者能领先对手多少分」：

        dp[i][j] = max( piles[i] - dp[i+1][j], piles[j] - dp[i][j-1] )

    区别只在于题目要求返回「先手是否获胜」，即 dp[0][n-1] >= 0（能保证不输即算先手胜，
    堆数为偶数时这个值恒非负，所以理论上直接返回 True 也对；这里保留 DP 给出通用解法，
    并与「预测赢家」对照）。

    为什么这题还有个「一定先手赢」的结论：当堆数 n 为偶数时，先手可以把所有石子按
    奇偶下标分成两堆，然后始终只取较多的一边——因为棋盘两端下标的奇偶性在每回合后都会
    互换，先手可以强制自己一直吃同一种奇偶下标，从而拿到两者中较大的那组，稳赢。
    本题保证 n 为偶数，故理论上直接返回 True 也对；这里保留 DP 是为了给出通用解法，并
    与「预测赢家」对照。

复杂度：时间 O(n^2)，空间 O(n^2)。
"""


def stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0


if __name__ == "__main__":
    assert stone_game([5, 3, 4, 5]) is True
    assert stone_game([3, 7, 2, 3]) is True
    assert stone_game([7, 7, 7, 7]) is True
    assert stone_game([1, 2]) is True
    print("stone_game: all tests passed")
