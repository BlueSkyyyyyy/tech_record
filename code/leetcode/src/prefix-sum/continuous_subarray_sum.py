"""523. 连续的子数组和（Continuous Subarray Sum）

题目：给定整数数组 nums 和整数 k，判断是否存在长度至少为 2 的连续子数组，
      其元素之和是 k 的整数倍（即被 k 整除）。
      例如 nums = [23, 2, 4, 6, 7]，k = 6，返回 True（[2, 4] 的和为 6）。

思路：与 974 同一个「同余前缀和」框架，只多了两个约束：子数组长度至少为 2，
      而且只要判断「存在」、不需要计数。
      子数组 nums[j+1..i] 的和为 prefix[i] - prefix[j]，要它被 k 整除，
      就要 prefix[i] ≡ prefix[j] (mod k)。于是我们仍然只看「前缀和 mod k」这个余数。
      区别在于：974 数个数，所以记录每个余数出现「几次」；本题要判断长度，
      所以记录每个余数「最早」出现的下标。扫描到下标 i、发现余数之前出现过最早在 j，
      那么存在一段 nums[j+1..i]，长度是 i - j，只要它 >= 2 就返回 True。
      为什么记最早下标而不是随便一个：固定右端 i 时，左端越早、长度越长，越容易满足长度限制，
      所以只需要最早的 j；用最早 j 求出的 i-j 是最大的可能长度。
      为什么初始 map 放 {0: -1}：空前缀余数为 0 且下标为 -1，这样从下标 0 开始的子数组
      长度为 i(-1) = i+1，公式统一。

复杂度：时间 O(n)，空间 O(min(n, k))。
"""


def check_subarray_sum(nums, k):
    first = {0: -1}
    prefix = 0
    for i, x in enumerate(nums):
        prefix = (prefix + x) % k
        if prefix in first:
            if i - first[prefix] >= 2:
                return True
        else:
            first[prefix] = i
    return False


if __name__ == "__main__":
    assert check_subarray_sum([23, 2, 4, 6, 7], 6) is True
    assert check_subarray_sum([23, 2, 6, 4, 7], 6) is True
    assert check_subarray_sum([23, 2, 6, 4, 7], 13) is False
    assert check_subarray_sum([5, 0, 0, 0], 3) is True
    assert check_subarray_sum([1, 0], 2) is False
    print("continuous_subarray_sum: all tests passed")
