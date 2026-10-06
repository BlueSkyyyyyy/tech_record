"""1649. 通过指令创建有序数组（Create Sorted Array through Instructions）

题目：给定整数数组 instructions，从左到右依次把每个数插入一个有序数组。每次插入的
    代价 = 新元素左边严格小于它的个数、右边严格大于它的个数，这两者的较小值。
    求所有插入代价之和对 1e9+7 取模的结果（即插入时数组中已有的元素里，比它小的个数
    与比它大的个数取较小者）。

思路（值域树状数组）：
    插入顺序固定，插入第 i 个数时，数组里恰好是前 i 个数。要算的「比 x 小的个数」
    与「比 x 大的个数」，正是「已出现过且 < x」和「已出现过且 > x」。用值域树状数组
    维护每个数值的出现次数：
      - less = prefix(x - 1)；
      - greater = processed - prefix(x)（processed 是已插入总数，prefix(x) 含等于 x 的）。
    取 min 累加，再把 x 插入。

    本题值域是 1..10^5，可以直接开满；若值域很大，就按前文先离散化。

复杂度：时间 O(n log M)（M = 值域上界），空间 O(M)。
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


def create_sorted_array(instructions):
    MOD = 10 ** 9 + 7
    MAXV = 100000
    bit = Fenwick(MAXV)

    ans = 0
    processed = 0
    for x in instructions:
        less = bit.prefix(x - 1)
        greater = processed - bit.prefix(x)
        ans = (ans + min(less, greater)) % MOD
        bit.add(x, 1)
        processed += 1
    return ans


if __name__ == "__main__":
    assert create_sorted_array([1, 5, 6, 2]) == 1
    assert create_sorted_array([1, 2, 3, 6, 5, 4]) == 3
    assert create_sorted_array([1, 2, 3, 4]) == 0
    assert create_sorted_array([4, 3, 2, 1]) == 0
    assert create_sorted_array([5]) == 0

    print("create_sorted_array: all tests passed")
