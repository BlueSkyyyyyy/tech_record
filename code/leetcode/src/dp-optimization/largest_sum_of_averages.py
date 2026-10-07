"""813. 最大平均值和的分组（Largest Sum of Averages）

题目：把数组 nums 分割成最多 k 个非空连续子数组，使各子数组「平均值之和」最大。

思路（DP + 前缀和优化）：
    设 dp[j][i] = 把前 i 个数分成 j 段的最大平均值和。枚举最后一段的起点 m：
        dp[j][i] = max_{m} ( dp[j-1][m] + (pre[i] - pre[m]) / (i - m) )
    其中 pre 是前缀和，这样任意一段的和、平均值都能 O(1) 算出。
    边界：dp[1][i] = pre[i] / i（只分一段就是整体平均）。
    答案 dp[k][n]。虽然整体仍是 O(k·n²)，但前缀和把转移里的区间求和从 O(n) 降到 O(1)。

复杂度：时间 O(k·n²)，空间 O(k·n)。
"""


def largest_sum_of_averages(nums, k):
    n = len(nums)
    pre = [0] * (n + 1)
    for i, v in enumerate(nums):
        pre[i + 1] = pre[i] + v

    dp = [[0.0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        dp[1][i] = pre[i] / i
    for j in range(2, k + 1):
        for i in range(j, n + 1):
            best = 0.0
            for m in range(j - 1, i):
                cand = dp[j - 1][m] + (pre[i] - pre[m]) / (i - m)
                if cand > best:
                    best = cand
            dp[j][i] = best
    return dp[k][n]


if __name__ == "__main__":
    assert abs(largest_sum_of_averages([9, 1, 2, 3, 9], 3) - 20.0) < 1e-9
    assert abs(largest_sum_of_averages([1, 2, 3, 4, 5, 6, 7], 4) - 20.5) < 1e-9
    assert abs(largest_sum_of_averages([5], 1) - 5.0) < 1e-9
    print("largest_sum_of_averages: all tests passed")
