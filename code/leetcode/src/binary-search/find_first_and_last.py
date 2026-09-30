"""34. 在排序数组中查找元素的第一个和最后一个位置（Find First and Last Position）

题目：给定一个非递减排列的整数数组 nums（可能含重复元素）和目标值 target，
      找出 target 在数组中出现的起始下标和结束下标；不存在则返回 [-1, -1]。
      要求时间复杂度 O(log n)。例如 nums = [5, 7, 7, 8, 8, 10]，target = 8 返回 [3, 4]，
      target = 6 返回 [-1, -1]。

思路：普通二分只能找到「某一个」target，落在重复段中间，拿不到边界。要拿到左右边界，
      需要两个专门的二分：
        - lower_bound(nums, target)：第一个 >= target 的下标；
        - upper_bound(nums, target)：第一个 > target 的下标。
      那么 target 存在的区间就是 [lower_bound, upper_bound - 1]。
      先求 lower_bound，如果它越界、或落到的元素不等于 target，说明 target 不存在，直接返回 [-1, -1]；
      否则再求 upper_bound，右边界就是它减一。

      为什么不能靠「找到 target 后往左右线性扩展」：重复元素可能很多，那一步是 O(n)，
      会退化掉二分的 O(log n)。用两次二分分别定位左右边界，仍是 O(log n)。

      为什么上下界都写成「先排除小于等于（或小于）的一段」：
      - lower_bound 判定 nums[mid] < target 就往右走，否则往左压，收敛到第一个 >= 的位置；
      - upper_bound 判定 nums[mid] <= target 就往右走，否则往左压，收敛到第一个 > 的位置。
      两者只差一个等号，是同一个模板换了判定条件。

复杂度：时间 O(log n)，空间 O(1)。
"""


def lower_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def upper_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def search_range(nums, target):
    first = lower_bound(nums, target)
    if first == len(nums) or nums[first] != target:
        return [-1, -1]
    last = upper_bound(nums, target) - 1
    return [first, last]


if __name__ == "__main__":
    assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    assert search_range([], 0) == [-1, -1]
    assert search_range([1], 1) == [0, 0]
    assert search_range([1], 0) == [-1, -1]
    assert search_range([2, 2, 2, 2], 2) == [0, 3]
    assert search_range([1, 2, 3], 2) == [1, 1]
    assert search_range([1, 2, 3], 4) == [-1, -1]
    print("find_first_and_last: all tests passed")
