"""238. 除自身以外数组的乘积（Product of Array Except Self）

题目：给定整数数组 nums，返回数组 answer，其中 answer[i] 等于 nums 中除 nums[i] 之外
      其余各元素的乘积。要求不使用除法，且在 O(n) 时间内完成。
      例如 nums = [1, 2, 3, 4]，返回 [24, 12, 8, 6]。

思路：不做除法，就要为每个位置分别算出「左边所有元素的积」和「右边所有元素的积」，再相乘。
      先用一次从左到右的遍历，把 res[i] 填成「i 左边所有元素的乘积」；
      再用一次从右到左的遍历，用一个 right 变量累计「i 右边所有元素的乘积」，
      直接乘进 res[i]。两趟遍历各 O(n)，合起来 O(n)。
      为什么可以省掉两个辅助数组：第一趟的左侧积已经直接存进 res，第二趟只需要一个滚动变量
      维护右侧积，节省了 O(n) 空间。
      为什么不能用除法：题目明确禁止，而且数组里可能含 0，除法还要处理 0 的个数，得不偿失。

复杂度：时间 O(n)，空间 O(1)（不计返回数组）。
"""


def product_except_self(nums):
    n = len(nums)
    res = [1] * n
    left = 1
    for i in range(n):
        res[i] = left
        left *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        res[i] *= right
        right *= nums[i]
    return res


if __name__ == "__main__":
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    assert product_except_self([2, 3]) == [3, 2]
    assert product_except_self([0, 0]) == [0, 0]
    assert product_except_self([5]) == [1]
    print("product_of_array_except_self: all tests passed")
