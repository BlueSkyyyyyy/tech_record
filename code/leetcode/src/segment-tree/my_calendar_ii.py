"""731. 我的日程安排表 II（My Calendar II）

题目：与「日程表 I」类似，但允许同一个时间点最多被两个日程覆盖；如果某个新日程会让
    某点出现「三重预订」（三个日程同时覆盖），则拒绝该日程。

思路（动态开点线段树 · 区间加 + 区间最大值）：
    和 729 同一棵树，只是判定阈值从 1 变成 2：新日程会给覆盖它的每个点 +1，
    所以只要 [start, end-1] 上当前最大值已经 >= 2，加入后就会出现 >= 3，必须拒绝。
    否则区间 +1 并接受。

    对照 729：`>= 1` 拒绝的是「任何重叠」，`>= 2` 拒绝的是「第三重重叠」。
    再进一步（见 732）连拒绝都不需要，直接返回加入后的全局最大重叠数。

复杂度：每次 book O(log C)，C = 1e9；空间 O(q log C)。
"""


class SegTree:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.lc = [0, 0]
        self.rc = [0, 0]
        self.mx = [0, 0]
        self.lz = [0, 0]

    def _new(self):
        self.lc.append(0)
        self.rc.append(0)
        self.mx.append(0)
        self.lz.append(0)
        return len(self.mx) - 1

    def _apply(self, o, v):
        self.mx[o] += v
        self.lz[o] += v

    def _push(self, o, l, r):
        if l >= r:
            return
        if not self.lc[o]:
            self.lc[o] = self._new()
        if not self.rc[o]:
            self.rc[o] = self._new()
        v = self.lz[o]
        if v:
            self._apply(self.lc[o], v)
            self._apply(self.rc[o], v)
            self.lz[o] = 0

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, v)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._update(self.lc[o], l, m, ql, qr, v)
        if qr > m:
            self._update(self.rc[o], m + 1, r, ql, qr, v)
        self.mx[o] = max(self.mx[self.lc[o]], self.mx[self.rc[o]])

    def add(self, ql, qr, v):
        if ql <= qr:
            self._update(1, self.lo, self.hi, ql, qr, v)

    def _query(self, o, l, r, ql, qr):
        if not o:
            return 0
        if ql <= l and r <= qr:
            return self.mx[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res = max(res, self._query(self.lc[o], l, m, ql, qr))
        if qr > m:
            res = max(res, self._query(self.rc[o], m + 1, r, ql, qr))
        return res

    def query(self, ql, qr):
        if ql > qr:
            return 0
        return self._query(1, self.lo, self.hi, ql, qr)


class MyCalendarTwo:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        if self.tree.query(start, end - 1) >= 2:
            return False
        self.tree.add(start, end - 1, 1)
        return True


if __name__ == "__main__":
    cal = MyCalendarTwo()
    assert cal.book(10, 20) is True
    assert cal.book(50, 60) is True
    assert cal.book(10, 40) is True
    assert cal.book(5, 15) is False
    assert cal.book(5, 10) is True
    assert cal.book(25, 55) is True
    print("my_calendar_ii: all tests passed")
