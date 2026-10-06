"""2426. 满足不等式的数对数目（Number of Pairs Satisfying Inequality）

题目：给定两个下标从 0 开始的整数数组 nums1、nums2 和整数 k，求满足
        i < j 且 nums1[i] - nums2[i] <= nums1[j] - nums2[j] + k
    的下标对 (i, j) 的数目。

思路（差值数组 + 值域树状数组）：
    令 diff[m] = nums1[m] - nums2[m]，不等式化成
        diff[i] - diff[j] <= k，
    即
        diff[i] <= diff[j] + k。
    于是问题变成：对每个 j，数它前面有多少个 diff 值不超过 diff[j] + k。

    从左往右扫描，用值域树状数组维护已出现的 diff 计数；扫到 diff[j] 时，求
    「<= diff[j] + k」的个数，直接把阈值离散化后用二分定位排名再前缀和即可。
    因为要按「值」比较，离散化前把所有 diff 值收集起来排序去重。

    这和 327 是同一副面孔：都是「前缀/差值 + 阈值区间计数」，差别只在阈值区间是
    单边（本题）还是双边（327）。

复杂度：时间 O(n log n)，空间 O(n)。
"""
from bisect import bisect_right


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


def number_of_pairs(nums1, nums2, k):
    n = len(nums1)
    diff = [nums1[i] - nums2[i] for i in range(n)]

    vals = sorted(set(diff))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    for j in range(n):
        threshold = diff[j] + k
        idx = bisect_right(vals, threshold)
        ans += bit.prefix(idx)
        bit.add(rank[diff[j]], 1)
    return ans


if __name__ == "__main__":
    assert number_of_pairs([3, 2, 5], [2, 2, 1], 1) == 3
    assert number_of_pairs([3, -1], [-2, 2], -1) == 0
    assert number_of_pairs([1, 2, 3], [3, 2, 1], 0) == 3
    assert number_of_pairs([1], [1], 0) == 0
    assert number_of_pairs([1, 1, 1], [1, 1, 1], 0) == 3

    print("number_of_pairs: all tests passed")
