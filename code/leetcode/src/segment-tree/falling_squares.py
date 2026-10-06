"""699. 掉落的方块（Falling Squares）

题目：在数轴上依次掉落方块，第 i 个方块占据 [left_i, left_i + side_i)，并落在它下方
    「已被占据的最大高度」之上。记录每次掉落后所有方块叠成的最高高度。

思路（坐标压缩 + 线段树区间赋值 / 区间最大值）：
    先把所有方块的左右端点收集起来排序去重，得到一串「坐标点」，相邻两点之间是一个
    高度恒定的区间。这样把 1e9 的坐标压到至多 2n 个点，线段树只在这些区间上工作。

    对每个方块：先查 [left, left+side) 内当前最大高度 base，它落在 base+side 处；
    然后把这个区间整体赋成 base+side（因为新方块比区间里原来的都高，赋值等价于取 max）。
    最后维护一个全局最大高度 cur = max(cur, base+side)，记入答案。

    为什么「赋值」够用：新高度 base+side 严格大于区间内所有旧高度，
    所以整块被新方块平整地抬高，直接用赋值覆盖即可，不需要再取 max。

复杂度：坐标压缩 O(n log n)；每个方块线段树操作 O(log n)；总 O(n log n)，空间 O(n)。
"""


class SegTree:
    def __init__(self, n):
        self.n = n
        self.mx = [0] * (4 * n)
        self.lz = [-1] * (4 * n)

    def _apply(self, o, v):
        self.mx[o] = v
        self.lz[o] = v

    def _push(self, o):
        if self.lz[o] != -1:
            self._apply(2 * o, self.lz[o])
            self._apply(2 * o + 1, self.lz[o])
            self.lz[o] = -1

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, v)
            return
        self._push(o)
        m = (l + r) // 2
        if ql <= m:
            self._update(2 * o, l, m, ql, qr, v)
        if qr > m:
            self._update(2 * o + 1, m + 1, r, ql, qr, v)
        self.mx[o] = max(self.mx[2 * o], self.mx[2 * o + 1])

    def _query(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.mx[o]
        self._push(o)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res = max(res, self._query(2 * o, l, m, ql, qr))
        if qr > m:
            res = max(res, self._query(2 * o + 1, m + 1, r, ql, qr))
        return res

    def assign(self, ql, qr, v):
        self._update(1, 0, self.n - 1, ql, qr, v)

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)


def falling_squares(positions):
    xs = sorted({x for left, side in positions for x in (left, left + side)})
    idx = {x: i for i, x in enumerate(xs)}
    st = SegTree(len(xs) - 1)
    res = []
    cur = 0
    for left, side in positions:
        li = idx[left]
        ri = idx[left + side] - 1
        base = st.query(li, ri)
        h = base + side
        st.assign(li, ri, h)
        cur = max(cur, h)
        res.append(cur)
    return res


if __name__ == "__main__":
    assert falling_squares([[1, 2], [2, 3], [6, 1]]) == [2, 5, 5]
    assert falling_squares([[100, 100], [200, 100]]) == [100, 100]
    assert falling_squares([[1, 1], [1, 1], [1, 1]]) == [1, 2, 3]
    print("falling_squares: all tests passed")
