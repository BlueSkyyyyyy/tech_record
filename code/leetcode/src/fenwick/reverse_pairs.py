"""493. 翻转对（Reverse Pairs）

题目：给定数组 nums，若 i < j 且 nums[i] > 2 * nums[j]，则称 (i, j) 是一个翻转对。
    返回翻转对的数量。

思路（值域树状数组，阈值 2x 也要离散化）：
    和逆序对同一模板：从左往右扫，维护「已出现过元素」的值域计数。扫到 x 时，要数
    「前面有多少个数 > 2x」，即 processed - prefix(rank(2x))。

    注意两点：
      1. 比较对象变成了 2 * nums[j]，所以离散化时要把 nums 和 2 * nums 两批值一起
         收进来排序去重，否则 2x 可能落不到某个排名上；
      2. 数字可能为负，2x 可能溢出 32 位，Python 无所谓，C++ 要用 long long。

    值域树状数组的计数：把所有出现过的值映射到排名后，prefix(r) 是「原始值不超过
    第 r 个排名的元素个数」，于是 > 2x 的个数用总数减掉即可。

复杂度：时间 O(n log n)，空间 O(n)。
"""


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


def reverse_pairs(nums):
    n = len(nums)
    if n == 0:
        return 0
    vals = sorted(set(nums) | {2 * x for x in nums})
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    processed = 0
    for x in nums:
        ans += processed - bit.prefix(rank[2 * x])
        bit.add(rank[x], 1)
        processed += 1
    return ans


if __name__ == "__main__":
    assert reverse_pairs([1, 3, 2, 3, 1]) == 2
    assert reverse_pairs([2, 4, 3, 5, 1]) == 3
    assert reverse_pairs([1, 2, 3, 4]) == 0
    assert reverse_pairs([-5, -5]) == 1
    assert reverse_pairs([5, -5]) == 1
    assert reverse_pairs([]) == 0

    print("reverse_pairs: all tests passed")
