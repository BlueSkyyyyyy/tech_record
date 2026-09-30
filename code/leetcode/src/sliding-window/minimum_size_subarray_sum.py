"""209. 长度最小的子数组（Minimum Size Subarray Sum）

题目：给定正整数数组 nums 和目标 target，找出**和 >= target** 的长度最小的连续子数组，返回其长度；
     不存在则返回 0。例如 target = 7、nums = [2,3,1,2,4,3]，答案是 [4,3]，长度 2。

思路：滑动窗口。right 不断右扩，把 nums[right] 加进窗口和 total；只要 total >= target，
     就尝试收缩：先用当前窗口长度更新答案，再从左侧吐出 nums[left] 并令 left 右移。
     收缩到 total < target 时停止，继续扩右端。

     为什么可以放心收缩左端：窗口内所有元素都是正数，吐出一个只会让和变小。
     只要 total >= target，继续移动 left 得到的窗口只会更短，正是我们要找的更优解。
     因此每条右端最多对应一次可行的左端扫描，整体 O(n)。

     为什么不能用这道题的思路解 560「和为 K 的子数组」：那题数组可含负数，
     窗口和不再单调，收缩条件失去依据；这里「正整数」是滑动窗口成立的前提。

复杂度：时间 O(n)，空间 O(1)。
"""


def min_sub_array_len(target, nums):
    left = 0
    total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best


if __name__ == "__main__":
    assert min_sub_array_len(7, [2, 3, 1, 2, 4, 3]) == 2
    assert min_sub_array_len(4, [1, 4, 4]) == 1
    assert min_sub_array_len(11, [1, 1, 1, 1, 1, 1, 1, 1]) == 0
    assert min_sub_array_len(15, [5, 1, 3, 5, 10, 7, 4, 9, 2, 8]) == 2
    assert min_sub_array_len(1, [1]) == 1
    print("minimum_size_subarray_sum: all tests passed")
