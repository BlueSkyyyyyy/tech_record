"""1025. 除数博弈（Divisor Game）

题目：初始数字 n。两人轮流操作：选一个能整除当前数字 n 的 x（满足 0 < x < n），把 n
换成 n - x。谁无法操作谁就输。你先手，问能否获胜。

思路（从「必胜 / 必败」反向递推）：
    用 dp[i] 表示「当前数字是 i 时，轮到行动的人是否能赢」。这是一个纯粹的状态博弈：
    当且仅当存在一种操作，让我走完之后对手面对的局面是「必败」，我才能赢。于是

        dp[i] = any( i % x == 0 and not dp[i - x]  for 1 <= x < i )

    边界：dp[1] 没有任何合法 x，行动者无法操作，故 dp[1] = False（必败）。

    为什么这题还能一眼看出答案：对任意偶数 i，取 x = 1，则 i - 1 是奇数。可以证明所有
    奇数都是必败态，所以偶数必胜；而奇数 i 的任意因子都是奇数，i - x 必为偶数（必胜态），
    故奇数必败。于是答案就是 n 是否为偶数。这里仍给出递推解，因为它完整展示了博弈题的
    通用套路。

复杂度：时间 O(n^2)，空间 O(n)。
"""


def divisor_game(n):
    dp = [False] * (n + 1)
    for i in range(2, n + 1):
        for x in range(1, i):
            if i % x == 0 and not dp[i - x]:
                dp[i] = True
                break
    return dp[n]


if __name__ == "__main__":
    assert divisor_game(1) is False
    assert divisor_game(2) is True
    assert divisor_game(3) is False
    assert divisor_game(4) is True
    assert divisor_game(5) is False
    assert divisor_game(1000) is True
    for n in range(1, 60):
        assert divisor_game(n) == (n % 2 == 0)
    print("divisor_game: all tests passed")
