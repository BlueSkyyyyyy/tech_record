"""33. 搜索旋转排序数组（Search in Rotated Sorted Array）

题目：整数数组 nums 原本升序排列且元素互不相同，但在某个未知下标处被「旋转」了
      （例如 [0,1,2,4,5,6,7] 变成 [4,5,6,7,0,1,2]）。给定 target，返回它的下标，不存在返回 -1。
      要求 O(log n)。

思路：整个数组已经不全局有序，无法直接用标准二分。但它有一个关键性质：
      在任意位置切一刀，**左右两半中必有一半是完全有序的**（旋转只在一个点断了顺序）。
      于是每轮取中点 mid 后：
        - 若 nums[left] <= nums[mid]：左半段 [left, mid] 有序。看 target 是否落在这个有序区间的范围
          [nums[left], nums[mid]) 内：是则收缩到左半段，否则去右半段；
        - 否则：右半段 [mid, right] 有序，同理判断 target 是否落在 (nums[mid], nums[right]] 内。
      每次都能排除一半，复杂度仍是 O(log n)。

      为什么判定用 nums[left] <= nums[mid] 而不是 <：当 left == mid 时（区间只剩一个元素）也需要归入
      「左半有序」，用 <= 才能覆盖这个边界。

      为什么区间判断的端点是开是闭：nums[mid] 已经在循环开头比较过、若不等于 target 就必然不是答案，
      所以左半段判断写成 nums[left] <= target < nums[mid]（严格小于 mid）；右半段写成
      nums[mid] < target <= nums[right]。

复杂度：时间 O(log n)，空间 O(1)。
"""


def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


if __name__ == "__main__":
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated([1], 0) == -1
    assert search_rotated([1], 1) == 0
    assert search_rotated([1, 3], 3) == 1
    assert search_rotated([5, 1, 3], 5) == 0
    assert search_rotated([5, 1, 3], 3) == 2
    assert search_rotated([3, 1], 1) == 1
    print("search_in_rotated_sorted_array: all tests passed")
