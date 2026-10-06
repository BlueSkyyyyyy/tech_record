"""977. 有序数组的平方（Squares of a Sorted Array）

题目：给你一个按非递减顺序排序的整数数组 nums（可以含负数），返回每个数字的
      平方组成的新数组，要求同样按非递减顺序排列。

思路（对撞双指针，从结果末尾往前填）：
    平方会抹掉符号：越靠近两端的数，绝对值越大，平方也越大。数组已经有序，所以
    「绝对值最大」的元素一定在两端之一，不可能藏在中间。

    于是用左右两个指针 l、r 分别指向当前未处理区间的两端，每轮比较 nums[l]、nums[r]
    的绝对值，把较大者平方后从结果数组的**末尾**往前放，再让对应指针向内收一格。

    为什么从末尾往前填：两端拿到的都是「当前最大」的平方，最大的数应该排在结果最后，
    所以从后往前写正合适；这和一个普通的归并（从两头取大的放末尾）形状相同。

    为什么不能直接平方后再排序：那样是 O(n log n)，而双指针利用了「原数组有序」这条
    现成信息，只要 O(n)。
"""


def sorted_squares(nums):
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    k = n - 1
    while l <= r:
        if abs(nums[l]) > abs(nums[r]):
            res[k] = nums[l] * nums[l]
            l += 1
        else:
            res[k] = nums[r] * nums[r]
            r -= 1
        k -= 1
    return res


if __name__ == "__main__":
    assert sorted_squares([-4, -1, 0, 3, 10]) == [0, 1, 9, 16, 100]
    assert sorted_squares([-7, -3, 2, 3, 11]) == [4, 9, 9, 49, 121]
    assert sorted_squares([0]) == [0]
    assert sorted_squares([-3, -2, -1]) == [1, 4, 9]
    assert sorted_squares([1, 2, 3]) == [1, 4, 9]
    print("sorted_squares: all tests passed")
