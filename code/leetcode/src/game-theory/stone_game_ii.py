"""1140. 石子游戏 II（Stone Game II）

题目：若干堆石子排成一行 piles。Alice 先手，每次可以从**队首**连续拿走 X 堆
（1 <= X <= 2M，初始 M = 1），拿完后 M 更新为 max(M, X)。两人轮流取，取到不能取为止，
各自拿走的总石子数即为得分。两人最优策略，问先手 Alice 最多能拿多少石子。

思路（状态是「起点 + 倍数上限 M」的 DP）：
    先手能拿多少，取决于「从第 i 堆开始、当前上限为 M」时行动者能拿到的最多石子数，
    记为 f(i, M)。设 suffix[i] = piles[i] 之后所有石子之和。若这一手拿走前 X 堆，则
    当前行动者拿到 suffix[i] - suffix[i+X]，对手随后在 i+X 处以 M' = max(M, X) 行动，
    能拿到 f(i+X, M')；因为剩下的石子最终都会被两人分完，当前行动者拿到的总数就是

        f(i, M) = max over X in [1, 2M] of ( suffix[i] - f(i+X, max(M, X)) )

    （这里用到了「剩余石子总量 = 我拿的 + 对手拿的」，所以「我剩下的部分」= 剩余总量
    减去对手能拿的。）i 走到末尾时 f = 0。

    为什么 M 会单调不减：M 只增不减，说明越往后一次能拿的上限越大，这是本题模拟的关键。

复杂度：时间 O(n^3)，空间 O(n^2)，n 为堆数（n <= 100）。
"""

from functools import lru_cache


def stone_game_ii(piles):
    n = len(piles)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]

    @lru_cache(maxsize=None)
    def f(i, m):
        if i >= n:
            return 0
        best = 0
        for x in range(1, 2 * m + 1):
            if i + x > n:
                break
            best = max(best, suffix[i] - f(i + x, max(m, x)))
        return best

    return f(0, 1)


if __name__ == "__main__":
    assert stone_game_ii([2, 7, 9, 4, 4]) == 10
    assert stone_game_ii([1, 2, 3]) == 3
    assert stone_game_ii([1]) == 1
    assert stone_game_ii([1, 2, 3, 4, 5, 6]) == 10
    print("stone_game_ii: all tests passed")
