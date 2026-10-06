"""53. 最大子数组和（Maximum Subarray）

题目：给定一个整数数组 nums，找出一个具有最大和的连续子数组（至少包含一个元素），
返回其最大和。

思路（动态规划：以「我」结尾的最大和）：
    要求的是「连续」子数组，连续性提示我们把状态钉在区间的右端点。设

        dp[i] = 以 nums[i] 结尾的最大子数组和。

    这样定义的好处是：任何连续子数组都有唯一的结尾，所以只要枚举所有结尾、取
    dp[i] 的最大值，就覆盖了全部候选。对 dp[i]，要么把前面的最优结果接上、要么
    从自己重新开始：

        dp[i] = max(nums[i], dp[i-1] + nums[i])

    前者表示「前面的和为正，接上更划算」；后者表示「前面的和为负，不如从当前
    元素另起一段」。因为只看前一项，用一个变量 cur 滚动即可；全局最大和 best 在
    每个位置顺手更新。

    为什么不能只返回最后一个 dp：最大子数组可能在任何位置结束，dp 的每一项都只是
    「以它结尾」的最优，必须再取一次全局最大。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_subarray_sum(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


if __name__ == "__main__":
    assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray_sum([1]) == 1
    assert max_subarray_sum([5, 4, -1, 7, 8]) == 23
    assert max_subarray_sum([-1]) == -1
    assert max_subarray_sum([-3, -1, -2]) == -1
    print("maximum_subarray: all tests passed")
