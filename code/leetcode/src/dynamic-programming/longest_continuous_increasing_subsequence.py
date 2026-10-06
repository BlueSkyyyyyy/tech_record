"""674. 最长连续递增序列（Longest Continuous Increasing Subsequence）

题目：给定未经排序的整数数组 nums，找出最长且连续递增的子序列的长度。

思路（动态规划：以「我」结尾的连续长度）：
    和 300 相比，多了「连续」二字。连续性让状态更简单：

        dp[i] = 以 nums[i] 结尾的连续递增序列长度。

    转移只取决于紧挨着的前一个元素：

        dp[i] = dp[i-1] + 1   若 nums[i] > nums[i-1]
        dp[i] = 1             否则（递增在 i 处断开，从 i 重新开始）

    同样只依赖前一项，用一个滚动变量 cur 即可，best 每步更新。

    对比 300 体会「连续」的威力：不连续时，前面任何一个较小的元素都可能是前驱，
    所以要枚举 j < i；连续时前驱只有一个（前一个），复杂度从 O(n^2) 降到 O(n)。

复杂度：时间 O(n)，空间 O(1)。
"""


def find_length_of_lcis(nums):
    if not nums:
        return 0
    best = cur = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best


if __name__ == "__main__":
    assert find_length_of_lcis([1, 3, 5, 4, 7]) == 3
    assert find_length_of_lcis([2, 2, 2, 2, 2]) == 1
    assert find_length_of_lcis([1, 3, 5, 7]) == 4
    assert find_length_of_lcis([]) == 0
    assert find_length_of_lcis([1]) == 1
    print("longest_continuous_increasing_subsequence: all tests passed")
