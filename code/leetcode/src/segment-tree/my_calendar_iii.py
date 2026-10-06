"""732. 我的日程安排表 III（My Calendar III）

题目：每次加入一个半开区间 [start, end) 的日程，返回加入之后「同时进行的日程数的最大值」。
    即所有时间段里被覆盖次数最多的那个点的覆盖次数（也叫最大重叠数）。

思路（动态开点线段树 · 区间加 + 全局最大值）：
    直接维护「每个点的覆盖次数」，加入一个日程就是区间 [start, end-1] 整体 +1。
    线段树的根节点维护的正是整棵树的全局最大值，所以入完直接读根即可。
    这正是 729/731 的推广：它们只需要「加入前查一下会不会越界」，而 732 要的是加入后的峰值。

    为什么要「动态开点」：坐标到 1e9，静态树开不下；只有被访问到的区间才建节点，
    单次操作新增 O(log C) 个节点。

复杂度：每次 book O(log C)，C = 1e9；空间 O(q log C)。

对照：729 用阈值 1 判断是否重叠，731 用阈值 2 判断是否三重；
    732 不再拒绝，直接返回全局最大重叠数 k。三题共用同一棵树。
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

    def query(self, ql, qr):
        if ql > qr:
            return 0
        return self._query(1, self.lo, self.hi, ql, qr)

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


class MyCalendarThree:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        self.tree.add(start, end - 1, 1)
        return self.tree.mx[1]


if __name__ == "__main__":
    cal = MyCalendarThree()
    assert cal.book(10, 20) == 1
    assert cal.book(50, 60) == 1
    assert cal.book(10, 40) == 2
    assert cal.book(5, 15) == 3
    assert cal.book(5, 10) == 3
    assert cal.book(25, 55) == 3
    print("my_calendar_iii: all tests passed")
