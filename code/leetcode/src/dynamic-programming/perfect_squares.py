"""279. 完全平方数（Perfect Squares）

题目：给你一个整数 n，返回和为 n 的完全平方数的最少数量。完全平方数是一个整数，
其值等于另一个整数的平方，如 1、4、9、16。

思路（完全背包·求最小值）：
    把 1², 2², 3², ... 看成物品，物品可以取无限多次，要凑出容量 n，问最少取几件。
    这就是与 322 零钱兑换同型的完全背包，只是「硬币面额」固定成了平方数。

    设 dp[i] 表示凑出 i 所需的最少平方数个数。最后一件事物是某个平方数 j²：

        dp[i] = min(dp[i - j²] + 1)   （对所有 j² <= i 的 j）

    初始化 dp[0] = 0，其余先置为 i（用 i 个 1 一定能凑出，作为上界），再逐一取 min。

    为什么内层从 1 枚举到 sqrt(i)：等价于把「平方数物品」逐个尝试了一遍，与完全背包
    先物品后容量异曲同工。容量方向没有 0/1 背包的倒序讲究，因为每种平方数本就允许重复
    使用。

复杂度：时间 O(n√n)，空间 O(n)。
"""


def num_squares(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = i
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]


if __name__ == "__main__":
    assert num_squares(12) == 3
    assert num_squares(13) == 2
    assert num_squares(1) == 1
    assert num_squares(0) == 0
    assert num_squares(7) == 4
    print("perfect_squares: all tests passed")
