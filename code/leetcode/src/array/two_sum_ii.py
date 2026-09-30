"""167. 两数之和 II - 输入有序数组（Two Sum II - Input Array Is Sorted）

题目：给定一个下标从 1 开始、按非递减顺序排列的整数数组 numbers，
找出两个数满足相加之和等于目标数 target，返回这两个数的下标（1-indexed）。
每个输入只对应一个答案，且不能重复使用同一个元素。

思路：数组已排序，用「对撞双指针」。
    lo 指向最小，hi 指向最大：
      - 和 == target：找到答案；
      - 和 <  target：需要更大，lo 右移；
      - 和 >  target：需要更小，hi 左移。
    为什么能跳过：若 a[lo] + a[hi] < target，则 a[lo] 与任何 a[hi-1] 更小，
    都不可能凑出 target，故 lo 可以安全右移；hi 同理。

复杂度：时间 O(n)，空间 O(1)。
"""


def two_sum_sorted(numbers, target):
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s == target:
            return [lo + 1, hi + 1]
        if s < target:
            lo += 1
        else:
            hi -= 1
    return []


if __name__ == "__main__":
    assert two_sum_sorted([2, 7, 11, 15], 9) == [1, 2]
    assert two_sum_sorted([2, 3, 4], 6) == [1, 3]
    assert two_sum_sorted([-1, 0], -1) == [1, 2]
    assert two_sum_sorted([1, 2, 3, 4, 4, 9], 8) == [4, 5]
    print("two_sum_ii: all tests passed")
