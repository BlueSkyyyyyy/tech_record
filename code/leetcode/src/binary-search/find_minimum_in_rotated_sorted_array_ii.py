"""154. 寻找旋转排序数组中的最小值 II（Find Minimum in Rotated Sorted Array II）

题目：和 153 相同，但数组**允许包含重复元素**。返回最小元素。要求在含重复的情况下也尽量高效。

思路：沿用 153「和右端比」的框架，但重复元素带来了新情况：当 nums[mid] == nums[right] 时，
      无法判断最小值在 mid 的哪一侧。例如 [1,1,1,0,1] 的 mid 和 right 可能都是 1，
      最小值 0 在右侧；而 [1,0,1,1,1] 同样 mid/right 都是 1，最小值却在左侧。两边都可能，信息丢失。

      这时的安全做法是 right -= 1：既然 nums[right] 和 nums[mid] 相等，而我们要找的最小值即使等于
      这个值，mid 位置也已经保留了它（mid < right），把最右端这个重复元素丢掉不会漏解。
      这个操作最坏会退化成 O(n)（例如全是相同元素），但当元素互不相同时就是 153 的 O(log n)。

      为什么不能像 153 那样直接比较：== 时不缩小 right 就无法前进；而随意选一侧会出错，
      right -= 1 是唯一「不丢正确答案」的保守收缩。

复杂度：平均/最好 O(log n)，最坏（大量重复）O(n)，空间 O(1)。
"""


def find_min_ii(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]


if __name__ == "__main__":
    assert find_min_ii([1, 3, 5]) == 1
    assert find_min_ii([2, 2, 2, 0, 1]) == 0
    assert find_min_ii([1, 1, 1, 0, 1]) == 0
    assert find_min_ii([1, 0, 1, 1, 1]) == 0
    assert find_min_ii([3, 3, 1, 3]) == 1
    assert find_min_ii([1]) == 1
    assert find_min_ii([1, 1]) == 1
    assert find_min_ii([2, 2, 2, 2]) == 2
    print("find_minimum_in_rotated_sorted_array_ii: all tests passed")
