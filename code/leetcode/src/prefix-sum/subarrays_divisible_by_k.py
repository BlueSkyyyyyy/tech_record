"""974. 和可被 K 整除的子数组（Subarray Sums Divisible by K）

题目：给定整数数组 nums 和整数 k，返回元素之和可被 k 整除的（连续、非空）子数组的数目。
      例如 nums = [4, 5, 0, -2, -3, 1]，k = 5，返回 7。

思路：和 560（和为 K 的子数组）是同一框架。子数组 nums[j..i-1] 的和等于
      prefix[i] - prefix[j]；要求它被 k 整除，等价于
      (prefix[i] - prefix[j]) % k == 0，即 prefix[i] ≡ prefix[j] (mod k)。
      于是问题变成：扫描时对每个前缀和，数一数前面有多少个前缀和与它「同余」。
      用哈希表记录每个余数出现过的次数，扫描到某个余数时，把它的历史出现次数累加进答案，
      再把这个余数次数加一。初始 count[0] = 1 代表空前缀，让「从数组开头开始」的子数组也能被统计。
      为什么用余数而不是前缀和本身：被 k 整除只关心模 k 的结果，同一个余数的前缀和可以互相配对，
      哈希键从「前缀和」换成「前缀和 mod k」，键空间一下就变小了。
      为什么必须取非负余数：语言里负数取模结果可能是负的（C++），要先 (+k)%k 归一化，
      否则 -1 和 k-1 会被当成不同的余数，配对就漏了。

复杂度：时间 O(n)，空间 O(min(n, k))。
"""


def subarrays_divisible_by_k(nums, k):
    count = {0: 1}
    prefix = 0
    res = 0
    for x in nums:
        prefix = (prefix + x) % k
        res += count.get(prefix, 0)
        count[prefix] = count.get(prefix, 0) + 1
    return res


if __name__ == "__main__":
    assert subarrays_divisible_by_k([4, 5, 0, -2, -3, 1], 5) == 7
    assert subarrays_divisible_by_k([5], 9) == 0
    assert subarrays_divisible_by_k([-1, 2, 9], 2) == 2
    assert subarrays_divisible_by_k([0, 0], 1) == 3
    assert subarrays_divisible_by_k([1, 2, 3], 3) == 3
    print("subarrays_divisible_by_k: all tests passed")
