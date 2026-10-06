"""1434. 每个人戴不同帽子的方案数（Number of Ways to Wear Different
Hats to Each Other）

题目：有 n 个人和若干顶帽子（编号 1..40）。hats[i] 是第 i 个人喜欢的所有帽子编号。
给每个人戴一顶喜欢的帽子，要求任意两人帽子不同，求方案数（对 1e9+7 取模）。

思路（状态 = "已经戴好帽子的人的集合"）：
    人数 n ≤ 10，谁已经戴好了可以用一个 n 位掩码表示，状态数 ≤ 2^n。

    关键是把"帽子"放到外层来枚举：帽子一共有 40 顶，每顶帽子最多只能给一个人，这正好
    对应"每件物品只能用一次"的 0/1 背包。用 dp[mask] 表示"戴好的人集合为 mask"的方案数。

    枚举每一顶帽子 h：它可以不分配（方案数继承），也可以分配给一个还没戴帽子、且喜欢
    它的那个人（把新集合的方案数累加上来）。因为一顶帽子在一轮里只处理一次，不会出现
    同一顶帽子被两个人戴的情况。

    把"人"压进掩码、把"帽子"当作外层物品，是本篇里"状态压缩 + 0/1 背包"的典型组合。

复杂度：时间 O(40 * 2^n * n)，空间 O(2^n)。
"""


def number_ways(hats):
    MOD = 10 ** 9 + 7
    n = len(hats)

    # 每顶帽子被哪些人喜欢（先对每个人的列表去重）
    likers = [[] for _ in range(41)]
    for i in range(n):
        for h in set(hats[i]):
            likers[h].append(i)

    dp = [0] * (1 << n)
    dp[0] = 1
    for h in range(1, 41):
        if not likers[h]:
            continue
        nxt = dp[:]  # 这顶帽子不分配，方案数原样保留
        for mask in range(1 << n):
            if dp[mask] == 0:
                continue
            for i in likers[h]:
                if not (mask >> i & 1):
                    nxt[mask | (1 << i)] = (nxt[mask | (1 << i)] + dp[mask]) % MOD
        dp = nxt
    return dp[(1 << n) - 1]


if __name__ == "__main__":
    assert number_ways([[3, 4], [4, 5], [5]]) == 1
    assert number_ways([[3, 5, 1], [3, 5]]) == 4
    assert number_ways([[1, 2, 3, 4]] * 4) == 24
    assert number_ways([[1], [1]]) == 0
    print("number_ways: all tests passed")
