"""315. 计算右侧小于当前元素的个数（Count of Smaller Numbers After Self）

题目：给定整数数组 nums，返回一个新数组 counts，其中 counts[i] 是 nums[i] 右侧
    严格小于 nums[i] 的元素个数。

思路（从右往左扫的值域树状数组）：
    问题等价于「对每个 i，数它右边有多少个更小的数」。如果我们**从右往左**扫描，
    那么「已经插入树状数组的数」恰好就是「当前元素的右侧元素」：
      - 插入的是数值 x 的排名，树状数组按值域计数；
      - 当前元素的答案 = 已插入且 < x 的个数 = prefix(rank(x) - 1)；
      - 记完答案再把 x 插进去。
    这与 LCR 170 的逆序对是同一套动作，只是方向相反、要保留每个位置的答案。

    离散化后树状数组大小为「不同取值个数」，即使数字很大或为负也适用。

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


def count_smaller(nums):
    n = len(nums)
    if n == 0:
        return []
    sorted_vals = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_vals)}
    bit = Fenwick(len(sorted_vals))

    ans = [0] * n
    for i in range(n - 1, -1, -1):
        r = rank[nums[i]]
        ans[i] = bit.prefix(r - 1)
        bit.add(r, 1)
    return ans


if __name__ == "__main__":
    assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_smaller([-1, -1]) == [0, 0]
    assert count_smaller([3, 4, 9, 6, 1]) == [1, 1, 2, 1, 0]
    assert count_smaller([1]) == [0]
    assert count_smaller([]) == []
    assert count_smaller([2, 0, 1]) == [2, 0, 0]

    print("count_smaller: all tests passed")
