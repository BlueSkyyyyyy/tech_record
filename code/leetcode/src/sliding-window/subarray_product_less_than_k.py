"""713. 乘积小于 K 的子数组（Subarray Product Less Than K）

题目：给定正整数数组 nums 和整数 k，返回乘积严格小于 k 的连续子数组的个数。
      例如 nums = [10,5,2,6]、k = 100，答案是 8。

思路：变长滑动窗口。仍然用 [left, right] 维护一个「乘积 < k」的窗口：
      右端进来 x 后若乘积 >= k，就不断从左边吐出元素直到乘积 < k。

      关键在计数方式：当窗口 [left, right] 合法时，**以 right 结尾**的合法子数组恰好有
      right - left + 1 个（左端可取 left..right 的任意位置，取更靠右的左端乘积一定更小）。
      把这些个数累加即可，无需再枚举子数组。

      为什么用乘法而不是和：本题统计的是乘积。< k 的约束下，元素都是正整数，窗口越长乘积越大，
      收缩依然单调有效。若 k <= 1，任何正整数乘积都 >= 1，直接返回 0（也避免除零边界问题）。

      为什么左端可以放心吐：窗口内全是正整数，吐出一个元素乘积只会变小或不变，所以一旦
      prod < k 就停下，此时是最靠右（最短）的合法左端，对应 right-left+1 个合法子数组。

复杂度：时间 O(n)，空间 O(1)。
"""


def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0
    prod = 1
    left = 0
    count = 0
    for right, x in enumerate(nums):
        prod *= x
        while prod >= k:
            prod //= nums[left]
            left += 1
        count += right - left + 1
    return count


if __name__ == "__main__":
    assert num_subarray_product_less_than_k([10, 5, 2, 6], 100) == 8
    assert num_subarray_product_less_than_k([1, 2, 3], 0) == 0
    assert num_subarray_product_less_than_k([1, 1, 1], 2) == 6
    assert num_subarray_product_less_than_k([1, 2, 3], 1) == 0
    assert num_subarray_product_less_than_k([2, 3], 6) == 2
    print("subarray_product_less_than_k: all tests passed")
