"""1218. 最长定差子序列（Longest Arithmetic Subsequence of Given Difference）

题目：给定数组 arr 和整数 difference，求最长的等差子序列（相邻两项差恰为 difference）的长度。

思路（DP 用哈希表把「找前驱」降到 O(1)）：
    朴素想法：dp[i] = 以 arr[i] 结尾的最长定差子序列长度，转移要往前找值等于
    arr[i] - difference 的 j，是 O(n²)。
    但「前驱」只由**值**决定，与下标无关：用字典 dp 记录「以某个值为结尾」的最长长度，
    扫描时直接查 dp[x - difference]：
        dp[x] = dp.get(x - difference, 0) + 1
    字典以值而不是下标为键，天然把重复值合并（取最长），一次遍历即可。

复杂度：时间 O(n)，空间 O(n)。
"""


def longest_subsequence(arr, difference):
    dp = {}
    best = 0
    for x in arr:
        dp[x] = dp.get(x - difference, 0) + 1
        best = max(best, dp[x])
    return best


if __name__ == "__main__":
    assert longest_subsequence([1, 2, 3, 4], 1) == 4
    assert longest_subsequence([1, 3, 5, 7], 1) == 1
    assert longest_subsequence([1, 5, 7, 8, 5, 3, 4, 2, 1], -2) == 4
    assert longest_subsequence([7], 0) == 1
    print("longest_arithmetic_subsequence: all tests passed")
