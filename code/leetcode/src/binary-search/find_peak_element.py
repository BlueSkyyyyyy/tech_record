"""162. 寻找峰值（Find Peak Element）

题目：峰值元素是指其值严格大于左右相邻元素的元素。给定整数数组 nums，数组可能包含多个峰值，
      返回任意一个峰值的下标。约定 nums[-1] = nums[n] = -∞。要求 O(log n)。

思路：峰值问题看起来和「有序」无关，但二分仍然成立，靠的是一种**局部单调性**：
      比较 nums[mid] 与它右边邻居 nums[mid+1]：
        - nums[mid] < nums[mid+1]：中点处于上升坡，沿着这个方向（向右）一定能在右半段找到峰值，
          令 left = mid + 1；
        - nums[mid] > nums[mid+1]：中点处于下降坡，峰值在 mid 或其左边，令 right = mid。
      循环写成 while left < right，收敛到某个峰值下标。

      为什么「上升就向右一定能找到峰值」：从 mid 向右走，只要还在上升就继续走；一旦开始下降，
      刚才那个最高点就是峰值；如果一直升到数组末尾，因为 nums[n] = -∞，末尾元素就是一个峰值。
      所以「向更高的一侧走」保证有解，这就是二分的单调性来源。

      为什么不用考虑左边：我们只需要返回任意一个峰值，向任意一个上升方向走都必然撞到一个峰值，
      不必两边都看。

复杂度：时间 O(log n)，空间 O(1)。
"""


def find_peak_element(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left


if __name__ == "__main__":
    assert find_peak_element([1, 2, 3, 1]) == 2
    assert find_peak_element([1, 2, 1, 3, 5, 6, 4]) in (1, 5)
    assert find_peak_element([1]) == 0
    assert find_peak_element([1, 2]) == 1
    assert find_peak_element([2, 1]) == 0
    assert find_peak_element([3, 2, 1]) == 0
    assert find_peak_element([1, 2, 3]) == 2
    print("find_peak_element: all tests passed")
