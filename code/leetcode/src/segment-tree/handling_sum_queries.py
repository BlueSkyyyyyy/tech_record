"""2569. 更新数组后处理求和查询（Handling Sum Queries After Update）

题目：给定两个数组 nums1、nums2 和若干查询 queries：
    - [1, l, r]：把 nums1 的 [l, r] 每个元素 0<->1 翻转；
    - [2, p, 0]：对每个 i，执行 nums2[i] += nums1[i] * p；
    - [3, 0, 0]：返回 sum(nums2)。
    只需返回所有类型 3 查询的答案。

思路（懒标记线段树 · 区间翻转 + 区间和）：
    nums1 的翻转用一个支持「区间翻转」的线段树维护，节点存 1 的个数 s。
    翻转一个区间时，s 变成「区间长度 - s」，并把翻转标记 lazy 取反下传。

    关键观察：我们从不访问 nums2 的单个元素，只关心它的总和。而类型 2 会让
    总和增加 p * sum(nums1)（因为每个位置按当前的 nums1 值加权）。于是维护
    total = sum(nums2)：类型 2 时 total += p * 当前 sum(nums1)；类型 3 直接返回 total。
    nums2 里的历史值不会被后续翻转影响，所以不必回改。

    为什么这样对：类型 2 之后 nums1 再翻转，改变的是 nums1 未来用于加权的值，
    而已经加进 nums2 的部分不再变化；我们只是把「每次加多少」按当时的状态累加。

复杂度：每次查询 O(log n)，n 为数组长度；空间 O(n)。
"""


class SegTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.s = [0] * (4 * self.n)
        self.lz = [False] * (4 * self.n)
        self._build(1, 0, self.n - 1, nums)

    def _build(self, o, l, r, nums):
        if l == r:
            self.s[o] = nums[l]
            return
        m = (l + r) // 2
        self._build(2 * o, l, m, nums)
        self._build(2 * o + 1, m + 1, r, nums)
        self.s[o] = self.s[2 * o] + self.s[2 * o + 1]

    def _apply(self, o, l, r):
        self.s[o] = (r - l + 1) - self.s[o]
        self.lz[o] = not self.lz[o]

    def _push(self, o, l, r):
        if self.lz[o]:
            m = (l + r) // 2
            self._apply(2 * o, l, m)
            self._apply(2 * o + 1, m + 1, r)
            self.lz[o] = False

    def _flip(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            self._apply(o, l, r)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._flip(2 * o, l, m, ql, qr)
        if qr > m:
            self._flip(2 * o + 1, m + 1, r, ql, qr)
        self.s[o] = self.s[2 * o] + self.s[2 * o + 1]

    def flip(self, ql, qr):
        self._flip(1, 0, self.n - 1, ql, qr)

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.s[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res += self._query(2 * o, l, m, ql, qr)
        if qr > m:
            res += self._query(2 * o + 1, m + 1, r, ql, qr)
        return res


def handle_queries(nums1, nums2, queries):
    st = SegTree(nums1)
    total = sum(nums2)
    res = []
    for t, p, _q in queries:
        if t == 1:
            st.flip(p, _q)
        elif t == 2:
            total += p * st.query(0, len(nums1) - 1)
        else:
            res.append(total)
    return res


if __name__ == "__main__":
    ans = handle_queries([1, 0, 1], [0, 0, 0], [[1, 1, 1], [2, 1, 0], [3, 0, 0]])
    assert ans == [3]
    ans = handle_queries([1], [5], [[2, 0, 0], [3, 0, 0]])
    assert ans == [5]
    ans = handle_queries([1, 0, 1], [0, 0, 0], [[2, 1, 0], [3, 0, 0]])
    assert ans == [2]
    print("handling_sum_queries: all tests passed")
