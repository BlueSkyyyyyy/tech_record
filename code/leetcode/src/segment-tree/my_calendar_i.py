"""729. 我的日程安排表 I（My Calendar I）

题目：实现 MyCalendar.book(start, end)，若 [start, end) 与已预订的区间都不重叠则加入
    并返回 True，否则不加入并返回 False。

思路（动态开点线段树 · 区间加 + 区间最大值）：
    两个半开区间 [a, b) 与 [c, d) 相交 <=> 存在整数点被覆盖两次。于是「会不会重叠」
    等价于「[start, end-1] 上已有的最大覆盖次数是否 >= 1」。用一棵线段树维护每个点的
    覆盖次数，支持「区间 +1」和「区间最大值查询」。

    坐标最大到 1e9，不可能开满整棵静态树，所以用「动态开点」：节点用到时才创建，
    一次操作最多新增 O(log C) 个节点。区间加用懒标记 lazy 记录「整个区间还没下发的增量」。

    为什么查询区间是 [start, end-1]：题目区间是半开的 [start, end)，整数点 end 不属于它，
    所以真正被占用的点是 start ... end-1。左闭右闭写起来最直观。

复杂度：每次 book O(log C)，C = 1e9 坐标上界；空间 O(q log C)，q 为预订次数。
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


class MyCalendar:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        if self.tree.query(start, end - 1) >= 1:
            return False
        self.tree.add(start, end - 1, 1)
        return True


if __name__ == "__main__":
    cal = MyCalendar()
    assert cal.book(10, 20) is True
    assert cal.book(15, 25) is False
    assert cal.book(20, 30) is True

    cal2 = MyCalendar()
    assert cal2.book(0, 1) is True
    assert cal2.book(1, 2) is True
    assert cal2.book(0, 2) is False
    print("my_calendar_i: all tests passed")
