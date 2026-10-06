"""715. Range 模块（Range Module）

题目：设计一个跟踪「半开区间 [left, right) 是否被跟踪」的数据结构：
    - addRange(left, right)：把该区间加入跟踪；
    - queryRange(left, right)：区间是否被完整跟踪；
    - removeRange(left, right)：取消该区间的跟踪。

思路（动态开点线段树 · 区间赋值 + 覆盖计数）：
    每个整数点只有「被跟踪 / 没被跟踪」两种状态，三种操作都是「区间整体赋成 1 / 0」，
    查询则是「这个区间里的被跟踪点数是否等于区间长度」。这是比 729~732 更进一步的线段树：
    懒标记不再是「增量」，而是「赋值」（1 或 0），下传时直接覆盖孩子。

    因为坐标到 1e9，仍用动态开点；节点维护 cnt = 该区间里被跟踪的整数点个数。
    赋 1 时 cnt = 区间长度，赋 0 时 cnt = 0。

    为什么用「覆盖计数」而不直接存布尔：区间长度各不相同，父节点的 cnt 要由两个孩子相加，
    存计数才能直接合并。

复杂度：每次操作 O(log C)，C = 1e9；空间 O(q log C)。
"""


class RangeModule:
    def __init__(self):
        self.lo, self.hi = 1, 10 ** 9
        self.lc = [0, 0]
        self.rc = [0, 0]
        self.cnt = [0, 0]
        self.lz = [-1, -1]

    def _new(self):
        self.lc.append(0)
        self.rc.append(0)
        self.cnt.append(0)
        self.lz.append(-1)
        return len(self.cnt) - 1

    def _apply(self, o, l, r, v):
        self.cnt[o] = v * (r - l + 1)
        self.lz[o] = v

    def _push(self, o, l, r):
        if l >= r:
            return
        if not self.lc[o]:
            self.lc[o] = self._new()
        if not self.rc[o]:
            self.rc[o] = self._new()
        v = self.lz[o]
        if v != -1:
            m = (l + r) // 2
            self._apply(self.lc[o], l, m, v)
            self._apply(self.rc[o], m + 1, r, v)
            self.lz[o] = -1

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, l, r, v)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._update(self.lc[o], l, m, ql, qr, v)
        if qr > m:
            self._update(self.rc[o], m + 1, r, ql, qr, v)
        self.cnt[o] = self.cnt[self.lc[o]] + self.cnt[self.rc[o]]

    def _query(self, o, l, r, ql, qr):
        if not o:
            return 0
        if ql <= l and r <= qr:
            return self.cnt[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res += self._query(self.lc[o], l, m, ql, qr)
        if qr > m:
            res += self._query(self.rc[o], m + 1, r, ql, qr)
        return res

    def addRange(self, left, right):
        self._update(1, self.lo, self.hi, left, right - 1, 1)

    def removeRange(self, left, right):
        self._update(1, self.lo, self.hi, left, right - 1, 0)

    def queryRange(self, left, right):
        return self._query(1, self.lo, self.hi, left, right - 1) == right - left


if __name__ == "__main__":
    rm = RangeModule()
    rm.addRange(10, 20)
    rm.removeRange(14, 16)
    assert rm.queryRange(10, 14) is True
    assert rm.queryRange(14, 16) is False
    assert rm.queryRange(16, 17) is True
    assert rm.queryRange(17, 20) is True
    assert rm.queryRange(10, 20) is False
    rm.addRange(14, 16)
    assert rm.queryRange(10, 20) is True
    rm.removeRange(10, 20)
    assert rm.queryRange(10, 20) is False
    print("range_module: all tests passed")
