"""724. 寻找数组的中心下标（Find Pivot Index）

题目：给定整数数组 nums，找出「中心下标」——即该下标左侧所有元素之和等于右侧所有元素之和。
      若存在多个，返回最左边那个；不存在返回 -1。规定下标 0 左侧为空、和为 0；末尾下标右侧同理。
      例如 nums = [1, 7, 3, 6, 5, 6]，中心下标是 3（左侧 1+7+3=11，右侧 5+6=11）。

思路：直接对每个位置分别求左右两边之和是 O(n^2)。用前缀和的思路把它压到 O(n)：
      设 total 为整个数组的和，扫描时维护 left 表示「当前元素左边所有元素之和」。
      在下标 i 处，右侧元素之和 = total - left - nums[i]。
      若 left == total - left - nums[i]，说明 i 就是中心下标（从左往右第一个找到的即最左）。
      否则把 nums[i] 并入 left，继续往右。
      为什么从左往右扫就够了：题目要「最左边」的中心下标，第一个满足条件的位置自然就是答案。
      注意必须先判断再累加 left，否则会把当前元素算进左侧。

复杂度：时间 O(n)，空间 O(1)。
"""


def pivot_index(nums):
    total = sum(nums)
    left = 0
    for i, x in enumerate(nums):
        if left == total - left - x:
            return i
        left += x
    return -1


if __name__ == "__main__":
    assert pivot_index([1, 7, 3, 6, 5, 6]) == 3
    assert pivot_index([1, 2, 3]) == -1
    assert pivot_index([2, 1, -1]) == 0
    assert pivot_index([0, 0, 0]) == 0
    assert pivot_index([1]) == 0
    assert pivot_index([-1, -1, -1, -1, -1, 0]) == 2
    print("find_pivot_index: all tests passed")
