"""312. 戳气球（Burst Balloons）

题目：有 n 个气球排成一排，第 i 个气球上写着 nums[i]。戳破第 i 个气球可以
得到 nums[i-1] * nums[i] * nums[i+1] 枚硬币（越界的气球按 1 计算），戳破后
它与左右的气球变成相邻。求能获得硬币的最大数量。

思路（区间 DP：枚举「最后」被戳的气球）：
    正向想很难：戳一个气球会改变左右邻居，后续依赖被破坏。反过来想——
    **枚举区间里最后一个被戳破的气球 k**。到它被戳时，区间 (i, j) 内除了 k
    已经全没了，而两侧边界 i、j 还在（它们是被保留的哨兵），于是这一步的
    收益是 vals[i] * vals[k] * vals[j]，且左右两段互不影响，可以独立求最优。

    在数组两端各加一个值为 1 的虚拟气球，作为永不被戳的边界哨兵：

        dp[i][j] = max over k in (i, j) of
                   vals[i]*vals[k]*vals[j] + dp[i][k] + dp[k][j]

    dp[i][j] 表示「戳光开区间 (i, j) 内所有气球」的最大收益（i、j 不动）。
    按区间长度从小到大填表，答案 dp[0][n+1]（n 为原气球个数）。

    为什么「最后被戳」能让子问题独立：k 是最后戳的，说明戳 k 时它在 (i, j)
    里的所有邻居都已消失，因此只与边界 i、j 有关；而 k 左侧、右侧内部的戳法
    各自是一段更小的同型问题，互不跨越 k。

复杂度：时间 O(n³)，空间 O(n²)。
"""


def max_coins(nums):
    vals = [1] + list(nums) + [1]
    n = len(vals)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                gain = vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j]
                if gain > dp[i][j]:
                    dp[i][j] = gain
    return dp[0][n - 1]


if __name__ == "__main__":
    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10
    assert max_coins([1]) == 1
    assert max_coins([]) == 0
    print("burst_balloons: all tests passed")
