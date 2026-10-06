"""322. 零钱兑换（Coin Change）

题目：给你一个整数数组 coins 表示不同面额的硬币，以及一个整数 amount 表示总金额。
计算并返回可以凑成总金额所需的「最少硬币个数」。如果没有任何一种硬币组合能凑出
总金额，返回 -1。每种硬币的数量视为无限多。

思路（完全背包·求最小值）：
    设 dp[i] 表示「凑出金额 i 所需的最少硬币数」。这是「完全背包」的形状：物品
    （硬币）可以取无限多次，问装满（恰好凑出）某个容量的最小代价。

    对每个金额 i，最后一枚硬币必然是某种面额 coin，于是：

        dp[i] = min(dp[i - coin] + 1)   （对所有满足 coin <= i 的硬币）

    为什么这样不重不漏：任何凑出 i 的方案，其最后一枚硬币是确定的某个面额；去掉它
    就得到一个凑出 i-coin 的方案。枚举所有可能的最后一枚硬币，就覆盖了全部方案。

    初始化：dp[0] = 0（凑 0 元用 0 枚），其余初始化为一个「比任何可行解都大」的哨兵
    （这里用 amount + 1，因为最多也只会用 amount 枚 1 元硬币）。若终值仍是哨兵，
    说明凑不出来，返回 -1。

    遍历顺序（先物品后容量、容量正序）：
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    容量**正序**是为了允许同一枚硬币被重复使用（完全背包的特征）；若倒序就退化成
    每枚硬币只能用一次的 0/1 背包了。先物品后容量的顺序对「求最小值」没有影响，
    但对「数方案」却至关重要（见 518）。

复杂度：时间 O(amount × len(coins))，空间 O(amount)。
"""


def coin_change(coins, amount):
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return -1 if dp[amount] == INF else dp[amount]


if __name__ == "__main__":
    assert coin_change([1, 2, 5], 11) == 3
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0
    assert coin_change([1], 2) == 2
    assert coin_change([2, 5, 10, 1], 27) == 4
    print("coin_change: all tests passed")
