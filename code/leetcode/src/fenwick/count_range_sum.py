"""327. 区间和的个数（Count of Range Sum）

题目：给定整数数组 nums 和两个整数 lower、upper，求满足 lower <= 区间和 <= upper
    的区间 [i, j]（i <= j）的个数。

思路（前缀和 + 值域树状数组）：
    记前缀和 prefix[k] = nums[0] + ... + nums[k-1]，那么区间和 sum(i, j) 等于
    prefix[j + 1] - prefix[i]。条件变成：
        lower <= prefix[j] - prefix[i] <= upper
    移项整理为：
        prefix[j] - upper <= prefix[i] <= prefix[j] - lower
    也就是说：对每个前缀 prefix[j]，数它**前面**有多少个前缀落在一个区间里。

    从左往右扫描前缀，用值域树状数组维护「已经出现过的前缀和」的计数；扫到
    prefix[j] 时，查询落在 [prefix[j]-upper, prefix[j]-lower] 里的已出现前缀个数，
    就是以下标 j 结尾的合法区间数。

    查询「值在 [lo, hi] 内的个数」时，先把所有前缀和离散化排序；用二分分别定位
    < lo 的个数和 <= hi 的个数，再在树状数组上做两次前缀和相减。

    为什么前缀和数组要包含 prefix[0] = 0：长度从 0 算起，第一个区间 [0, j] 的
    左端对应 i = 0，即 prefix[0]。所以扫描前先把 prefix[0] 插入。

复杂度：时间 O(n log n)，空间 O(n)。
"""
from bisect import bisect_left, bisect_right


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


def count_range_sum(nums, lower, upper):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    vals = sorted(set(prefix))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    bit.add(rank[prefix[0]], 1)
    for j in range(1, n + 1):
        lo = prefix[j] - upper
        hi = prefix[j] - lower
        left = bisect_left(vals, lo)
        right = bisect_right(vals, hi)
        ans += bit.prefix(right) - bit.prefix(left)
        bit.add(rank[prefix[j]], 1)
    return ans


if __name__ == "__main__":
    assert count_range_sum([-2, 5, -1], -2, 2) == 3
    assert count_range_sum([0], 0, 0) == 1
    assert count_range_sum([0], 1, 1) == 0
    assert count_range_sum([0, 0, 0], 0, 0) == 6
    assert count_range_sum([-1, -1, -1], -1, 0) == 3
    assert count_range_sum([1, 2, 3], 3, 6) == 4

    print("count_range_sum: all tests passed")
