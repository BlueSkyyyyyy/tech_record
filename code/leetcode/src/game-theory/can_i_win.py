"""464. 我能赢吗（Can I Win）

题目：从 1 到 maxChoosableInteger 这些整数里，两人轮流选一个「还没被选过」的数，累加
到各自选出的数字之和上。谁先让累加和达到（或超过）desiredTotal 谁获胜。你先手，问
能否保证获胜。

思路（位掩码 + 记忆化搜索）：
    「已经选了哪些数」是唯一的状态，而它可以用一个整数的二进制位表示：第 i 位为 1 表示
    数字 i 已被选走。于是把整个博弈写成递归：

        win(mask, total):
            对每个还没用过的 i：
                若 total + i >= desiredTotal，直接赢
                否则若对手在 mask | (1<<i) 上必败，也赢
            一个能赢的走法都没有，就输

    为什么不用把 total 也放进记忆化键：total 完全由「选了哪些数」决定，即 total 是 mask
    的函数，所以只用 mask 当键即可，不会误伤。

    为什么需要位掩码：maxChoosableInteger 最大 20，用布尔数组或元组做状态在记忆化里开销
    大；压成整数的 20 个二进制位，既省空间又能直接当哈希键。

复杂度：时间 O(2^n * n)，空间 O(2^n)，n = maxChoosableInteger（n <= 20）。
"""


def can_i_win(max_choosable_integer, desired_total):
    if desired_total <= 0:
        return True
    total = max_choosable_integer * (max_choosable_integer + 1) // 2
    if total < desired_total:
        return False
    memo = {}

    def win(mask, total):
        if mask in memo:
            return memo[mask]
        for i in range(1, max_choosable_integer + 1):
            bit = 1 << (i - 1)
            if mask & bit:
                continue
            if total + i >= desired_total or not win(mask | bit, total + i):
                memo[mask] = True
                return True
        memo[mask] = False
        return False

    return win(0, 0)


if __name__ == "__main__":
    assert can_i_win(10, 0) is True
    assert can_i_win(10, 1) is True
    assert can_i_win(10, 11) is False
    assert can_i_win(1, 1) is True
    assert can_i_win(1, 2) is False
    assert can_i_win(2, 2) is True
    assert can_i_win(2, 3) is False
    assert can_i_win(3, 5) is True
    print("can_i_win: all tests passed")
