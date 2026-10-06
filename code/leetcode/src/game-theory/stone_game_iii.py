"""1406. 石子游戏 III（Stone Game III）

题目：若干堆石子排成一行 stoneValue。Alice 先手，每次可以从**队首**拿走 1、2 或 3 堆，
拿到的石子计入自己的总分。两人轮流取，直到取完。两人最优策略，问 Alice 的总分能否
严格大于 Bob（能则 Alice 胜）。

思路（以「分差」为状态的后缀 DP）：
    与 486 / 877 同样的分差思想，只是可选步长固定为 1、2、3。设 dp[i] 表示「从第 i 堆
    开始，轮到行动的人最终能领先对手的分数」。若这一手取前 x 堆（x = 1,2,3），当场得到
    suffix[i] - suffix[i+x]，而对手随后在 i+x 处以行动者身份能领先 dp[i+x]，故净分差为

        (suffix[i] - suffix[i+x]) - dp[i+x]

    对所有合法 x 取最大。dp[n] = 0。最后 dp[0] > 0 即 Alice 总分更高。

    为什么可以倒着推：dp[i] 只依赖更靠后的 dp[i+1..i+3]，从右往左一遍即可。

复杂度：时间 O(n)，空间 O(n)。
"""


def stone_game_iii(stone_value):
    n = len(stone_value)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + stone_value[i]

    dp = [0] * (n + 4)
    for i in range(n - 1, -1, -1):
        best = float("-inf")
        for x in (1, 2, 3):
            if i + x <= n:
                take = suffix[i] - suffix[i + x]
                best = max(best, take - dp[i + x])
        dp[i] = best
    return dp[0] > 0


if __name__ == "__main__":
    assert stone_game_iii([1, 2, 3, 7]) is False
    assert stone_game_iii([1, 2, 3, -9]) is True
    assert stone_game_iii([1, 2, 3, 6]) is False
    assert stone_game_iii([1, 2, 3]) is True
    assert stone_game_iii([1]) is True
    print("stone_game_iii: all tests passed")
