"""153. 寻找旋转排序数组中的最小值（Find Minimum in Rotated Sorted Array）

题目：一个原本升序、元素互不相同的数组在某个未知下标处被旋转（如 [3,4,5,1,2]）。
      返回数组中的最小元素。要求 O(log n)。

思路：旋转后的数组有一个「折点」，折点右侧就是最小值所在。我们用「中点和右端点比较」来判断
      最小值在中点左边还是右边：
        - nums[mid] > nums[right]：说明 mid 落在较大的左半段，最小值一定在 mid 右边（不含 mid），
          令 left = mid + 1；
        - nums[mid] < nums[right]：说明从 mid 到 right 是递增的，最小值在 mid 或其左边，令 right = mid。
      循环写成 while left < right（左闭右开式收缩），退出时 left == right，正好指向最小值。

      为什么和 nums[right] 比，而不是和 nums[left] 比：旋转后数组的「断点」信息在右端更稳定。
      和右端比，nums[mid] > nums[right] 一定意味着最小值在右侧；而和左端比在数组未旋转时会退化，
      逻辑更绕。固定「和右端比」是一套好记的模板。

      为什么 right = mid 而不是 mid - 1：nums[mid] < nums[right] 时，mid 本身可能就是最小值
      （例如 [2,1] 的 mid=0 时 nums[0]=2 > nums[1]，走左支；再如 [1,2] 的 mid=0 时 nums[0]<nums[1]，
      最小值就是 mid=0），所以 mid 必须留在候选区间里。

复杂度：时间 O(log n)，空间 O(1)。
"""


def find_min(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


if __name__ == "__main__":
    assert find_min([3, 4, 5, 1, 2]) == 1
    assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min([11, 13, 15, 17]) == 11
    assert find_min([1]) == 1
    assert find_min([2, 1]) == 1
    assert find_min([1, 2]) == 1
    assert find_min([5, 1, 2, 3, 4]) == 1
    print("find_minimum_in_rotated_sorted_array: all tests passed")
