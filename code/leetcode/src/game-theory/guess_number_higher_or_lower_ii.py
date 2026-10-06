"""375. 猜数字大小 II（Guess Number Higher or Lower II）

题目：我心里想一个 1~n 之间的数字。你每次猜一个数 x，如果猜错我会告诉你「大了 / 小了」，
并且你为这次猜测支付 x 元。无论我想的是哪个数字，你都能保证猜中，问最少要准备多少钱
（最坏情况下的花费）。

思路（minimax 区间 DP）：
    猜数会把区间切成两半：「猜 x」花掉 x，之后要么答案在左边 [i, x-1]、要么在右边
    [x+1, j]。由于对手（出题人）总把情况引向你更亏的那半边，最坏花费是两边代价的较大者。
    我要在 x 的所有选择里挑一个让最坏花费最小的：

        dp[i][j] = min over x in [i, j] of ( x + max(dp[i][x-1], dp[x+1][j]) )

    区间长度为 1 时一眼猜中不用花钱，dp[i][i] = 0；空区间 dp[i][i-1] 也记 0。

    为什么用「取 max」而不是「取平均」：这是零和博弈，你要的是**保证**能猜中，
    对手会挑最坏的分支；所以内层取 max（对抗），外层取 min（我方最优）。

复杂度：时间 O(n^3)，空间 O(n^2)（n <= 200，足够）。
"""


def get_money_amount(n):
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            best = float("inf")
            for x in range(i, j + 1):
                cost = x + max(dp[i][x - 1], dp[x + 1][j])
                if cost < best:
                    best = cost
            dp[i][j] = best
    return dp[1][n] if n >= 1 else 0


if __name__ == "__main__":
    assert get_money_amount(1) == 0
    assert get_money_amount(2) == 1
    assert get_money_amount(3) == 2
    assert get_money_amount(4) == 4
    assert get_money_amount(10) == 16
    print("guess_number_higher_or_lower_ii: all tests passed")
