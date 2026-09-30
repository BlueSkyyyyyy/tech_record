"""704. 二分查找（Binary Search）

题目：给定一个升序排列的整数数组 nums 和一个目标值 target，在数组中查找 target。
      如果存在返回它的下标，否则返回 -1。数组中的元素不重复（LeetCode 原题如此）。

思路：二分的本质是「每次排除一半的搜索范围」。维护一个闭区间 [left, right]，
      只要区间非空就取中点 mid = left + (right - left) // 2，比较 nums[mid] 与 target：
        - 相等：直接返回 mid；
        - nums[mid] < target：target 只可能在右半段，令 left = mid + 1；
        - nums[mid] > target：target 只可能在左半段，令 right = mid - 1。
      每轮区间长度至少减半，所以最多循环 O(log n) 次。

      为什么 mid 写成 left + (right - left) // 2 而不写 (left + right) // 2：
      后者在 C++ 里当 left、right 接近 int 上限时相加会溢出，前者先减后加可避免。

      为什么循环条件是 left <= right、更新时又带 ±1：
      区间是「闭」的，left == right 时区间里还剩一个元素，必须再比一次；
      排除 mid 时把它明确移出区间（±1），下一轮才不会重复考察同一个位置。
      若写成 left < right 或更新时不 ±1，就会出现漏查或死循环。

复杂度：时间 O(log n)，空间 O(1)。
"""


def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


if __name__ == "__main__":
    assert search([-1, 0, 3, 5, 9, 12], 9) == 4
    assert search([-1, 0, 3, 5, 9, 12], 2) == -1
    assert search([5], 5) == 0
    assert search([5], -5) == -1
    assert search([1, 2, 3, 4, 5], 1) == 0
    assert search([1, 2, 3, 4, 5], 5) == 4
    assert search([], 1) == -1
    print("binary_search: all tests passed")
