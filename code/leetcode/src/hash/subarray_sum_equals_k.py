"""560. 和为 K 的子数组（Subarray Sum Equals K）

题目：给定整数数组 nums 和整数 k，统计有多少个**连续子数组**的元素和恰好等于 k。

思路：暴力枚举所有子数组求和是 O(n^2)。用前缀和的视角加速：
     记 prefix[i] 为「前 i 个元素之和」，则子数组 nums[j..i-1] 的和 = prefix[i] - prefix[j]。
     要求它等于 k，即 prefix[j] == prefix[i] - k。
     于是边扫描边维护一张哈希表：prefix -> 该前缀和出现过的次数。
     扫描到位置 i 时，再加上「表里 prefix[i] - k 出现的次数」，就是以 i 结尾的合法子数组个数。
     关键初始化：count[0] = 1，代表空前缀。这样从下标 0 开始的子数组（prefix[j]=0，j=0）
     也能被统计到，否则会漏掉所有从头开始的解。
     为什么是哈希而不是双指针：数组可能含负数，前缀和不是单调的，窗口无法单向收缩。

复杂度：时间 O(n)，空间 O(n)。
"""


def subarray_sum(nums, k):
    count = {0: 1}
    prefix = 0
    total = 0
    for x in nums:
        prefix += x
        total += count.get(prefix - k, 0)
        count[prefix] = count.get(prefix, 0) + 1
    return total


if __name__ == "__main__":
    assert subarray_sum([1, 1, 1], 2) == 2
    assert subarray_sum([1, 2, 3], 3) == 2
    assert subarray_sum([1, -1, 0], 0) == 3
    assert subarray_sum([-1, -1, 1], 0) == 1
    print("subarray_sum_equals_k: all tests passed")
