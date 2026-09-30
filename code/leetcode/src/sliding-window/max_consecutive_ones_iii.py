"""1004. 最大连续 1 的个数 III（Max Consecutive Ones III）

题目：给定二进制数组 nums 和整数 k，最多可以把 k 个 0 翻成 1，求能得到的连续 1 的最大个数。
      例如 nums = [1,1,1,0,0,0,1,1,1,1,0]、k = 2，答案是 6。

思路：变长滑动窗口，和 424 是同一个模型。窗口内有多少个 0，就需要多少次翻转；
      只要 0 的个数 <= k，窗口就合法，窗口长度就是一段连续的 1。

      右端一路右扩，遇到 0 就把 zeros 加一；当 zeros > k 时收缩左端，若离开的是 0 则 zeros 减一，
      直到窗口重新合法。用窗口长度更新答案。

      为什么是同一模型：424 的「窗口长度 - 出现最多的字符数」本质是「需要改动的字符数」，
      本题「窗口内 0 的个数」就是「需要改动的字符数」，只是本题目标字符固定为 1，不必再求
      最多频次。把 424 里 max_freq 换成「当前窗口长度 - zeros」即可互相映射。

复杂度：时间 O(n)，空间 O(1)。
"""


def longest_ones(nums, k):
    left = 0
    zeros = 0
    best = 0
    for right, x in enumerate(nums):
        if x == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


if __name__ == "__main__":
    assert longest_ones([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2) == 6
    assert longest_ones([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3) == 10
    assert longest_ones([1, 1, 1], 0) == 3
    assert longest_ones([0, 0, 0], 0) == 0
    assert longest_ones([0, 0, 0], 3) == 3
    print("max_consecutive_ones_iii: all tests passed")
