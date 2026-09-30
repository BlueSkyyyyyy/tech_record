"""303. 区域和检索 - 数组不可变（Range Sum Query - Immutable）

题目：设计一个类 NumArray，用整数数组 nums 初始化，支持查询下标区间 [left, right] 内
      所有元素之和。数组在整个过程中不会被修改。
      例如 nums = [-2, 0, 3, -5, 2, -1]，sum_range(0, 2) = 1，sum_range(2, 5) = -1。

思路：如果每次查询都从 left 累加到 right，单次是 O(n)，查询多了会很慢。
      前缀和的核心是「预存快照」：先花一次 O(n) 建一个数组 prefix，
      其中 prefix[i] 表示「前 i 个元素之和」（即 nums[0..i-1]），特别地 prefix[0] = 0。
      于是区间和可以拼出来：
          sum(nums[left..right]) = prefix[right + 1] - prefix[left]
      为什么成立：prefix[right+1] 是从头加到 right，prefix[left] 是从头加到 left-1，
      两者相减，头部 [0, left-1] 被抵消，剩下的正好是 [left, right]。
      多留一个 prefix[0] = 0，是为了让 left = 0 时也能套用同一个公式，不必特判。

复杂度：预处理时间 O(n)、空间 O(n)；每次查询时间 O(1)。
"""


class NumArray:
    def __init__(self, nums):
        self.prefix = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            self.prefix[i + 1] = self.prefix[i] + x

    def sum_range(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]


if __name__ == "__main__":
    arr = NumArray([-2, 0, 3, -5, 2, -1])
    assert arr.sum_range(0, 2) == 1
    assert arr.sum_range(2, 5) == -1
    assert arr.sum_range(0, 5) == -3
    assert arr.sum_range(3, 3) == -5
    arr2 = NumArray([5])
    assert arr2.sum_range(0, 0) == 5
    print("range_sum_query: all tests passed")
