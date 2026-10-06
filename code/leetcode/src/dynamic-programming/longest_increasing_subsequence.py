"""300. 最长递增子序列（Longest Increasing Subsequence）

题目：给定整数数组 nums，找到其中最长严格递增子序列的长度。子序列不要求连续，
但元素的相对顺序不能改变。

思路（动态规划：以「我」结尾的最长长度）：
    子序列「可以不连续」，很难直接看出从哪转移。把它钉在结尾上就清楚了。设

        dp[i] = 以 nums[i] 结尾的最长递增子序列长度。

    那么 dp[i] 至少为 1（只有它自己）；只要前面存在某个比 nums[i] 小的 nums[j]，
    就可以把以 j 结尾的最优解接过来，再接上 nums[i]：

        dp[i] = 1 + max(dp[j])  对所有 j < i 且 nums[j] < nums[i]

    答案为 max(dp)。两层循环，外层从前往后保证 j 的状态已经算好。

    为什么是「严格」小于：题目要求严格递增，所以条件写 nums[j] < nums[i]；
    若允许非递减则改为 <=。

    存在 O(n log n) 的解法（tails + 二分：维护每个长度对应的最小结尾，用二分
    找第一个 >= 当前值的位置替换或追加），但那把「dp 以 i 结尾」的直觉藏进了
    数据结构里。作为序列 DP 的入门，这里详展 O(n^2) 的模板，二分版一句话带过。

复杂度：时间 O(n^2)，空间 O(n)。
"""


def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


if __name__ == "__main__":
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
    assert length_of_lis([7, 7, 7, 7, 7]) == 1
    assert length_of_lis([1, 2, 3, 4, 5]) == 5
    assert length_of_lis([]) == 0
    print("longest_increasing_subsequence: all tests passed")
