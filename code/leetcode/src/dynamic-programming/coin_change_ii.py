"""518. 零钱兑换 II（Coin Change II）

题目：给你一个整数数组 coins 表示不同面额的硬币，以及一个整数 amount 表示总金额。
计算并返回可以凑成总金额的「硬币组合数」。如果任何硬币组合都无法凑出总金额，返回 0。
每种硬币的数量视为无限多，且顺序不同的序列视为同一种组合。

思路（完全背包·数方案）：
    状态与 322 相同：dp[i] 表示「凑出金额 i 的方案数」。区别只在聚合运算符——322 求
    「最少几枚」取 min，这里求「有几种凑法」用加法：

        dp[i] += dp[i - coin]   （对所有满足 coin <= i 的硬币）

    初始化 dp[0] = 1：凑出 0 元有且只有一种方案——什么都不选。（注意是 1 不是 0，
    否则所有方案数都会是 0。）

    一个组合里每枚硬币只用一次「遍历机会」。若把硬币放在外层循环、金额放内层，同一
    组合中硬币出现的前后顺序就被固定为 coins 的下标顺序，因此「1+2」和「2+1」只会被
    算作一种，正好满足「不区分顺序」的要求。

    若反过来把金额放外层、硬币放内层（dp[i] += dp[i-coin] 对每个 i 都枚举硬币），
    就会把不同顺序当成不同组合，得到的是「排列数」，那是 377. 组合总和 IV 的答案。
    所以：**求组合数必须让物品在外层、容量在内层。**

复杂度：时间 O(amount × len(coins))，空间 O(amount)。
"""


def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]


if __name__ == "__main__":
    assert change(5, [1, 2, 5]) == 4
    assert change(3, [2]) == 0
    assert change(10, [10]) == 1
    assert change(0, []) == 1
    assert change(7, [1, 2, 5]) == 6
    print("coin_change_ii: all tests passed")
