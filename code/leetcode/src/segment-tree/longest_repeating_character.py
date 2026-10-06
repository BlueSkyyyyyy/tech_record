"""2213. 由单个字符重复的最长子字符串（Longest Substring of One Repeating Character）

题目：给定字符串 s，每次把下标 index 的字符改成 ch，问每次修改后「由单个字符重复形成的
    最长子串」的长度。

思路（线段树维护「区间合并信息」）：
    单点修改、每次要全局「最长同类连续段」，正适合线段树。关键是设计每个节点要维护的量，
    使两个相邻区间能合并出正确答案。每个节点存 5 个数（外加长度）：
      - len：区间长度；
      - lch, llen：最左端字符、以及从左端起连续相同字符的长度（左前缀）；
      - rch, rlen：最右端字符、以及从右端起连续相同字符的长度（右后缀）；
      - best：本区间内最长同类连续段。

    合并左右两段 a、b 时：
      - 左前缀：若 a 整段都是同一字符（llen == len）且 a.lch == b.lch，
        则能跨过边界延伸到 b 的左前缀，长度 = a.len + b.llen；否则就是 a.llen。
      - 右后缀：对称地，若 b 整段同色且 b.rch == a.rch，则为 b.len + a.rlen；否则 b.rlen。
      - best：先取 max(a.best, b.best)；若 a.rch == b.lch，还可以把「a 的右后缀 +
        b 的左前缀」拼起来跨边界，取 max。

    每次修改只更新一条叶子到根的路径，O(log n)，根节点的 best 就是答案。

复杂度：建树 O(n)，每次修改 O(log n)；空间 O(n)。
"""


class SegTree:
    def __init__(self, s):
        self.n = len(s)
        self.s = s
        N = 4 * self.n
        self.ln = [0] * N
        self.lc = [0] * N
        self.ll = [0] * N
        self.rc = [0] * N
        self.rl = [0] * N
        self.bs = [0] * N
        self._build(1, 0, self.n - 1)

    def _build(self, o, l, r):
        if l == r:
            c = self.s[l]
            self.ln[o] = 1
            self.lc[o] = c
            self.ll[o] = 1
            self.rc[o] = c
            self.rl[o] = 1
            self.bs[o] = 1
            return
        m = (l + r) // 2
        self._build(2 * o, l, m)
        self._build(2 * o + 1, m + 1, r)
        self._pull(o)

    def _pull(self, o):
        left, right = 2 * o, 2 * o + 1
        self.ln[o] = self.ln[left] + self.ln[right]

        self.lc[o] = self.lc[left]
        self.ll[o] = self.ll[left]
        if self.ll[left] == self.ln[left] and self.lc[left] == self.lc[right]:
            self.ll[o] = self.ln[left] + self.ll[right]

        self.rc[o] = self.rc[right]
        self.rl[o] = self.rl[right]
        if self.rl[right] == self.ln[right] and self.rc[right] == self.rc[left]:
            self.rl[o] = self.ln[right] + self.rl[left]

        best = max(self.bs[left], self.bs[right])
        if self.rc[left] == self.lc[right]:
            best = max(best, self.rl[left] + self.ll[right])
        self.bs[o] = best

    def update(self, idx, ch):
        self._update(1, 0, self.n - 1, idx, ch)

    def _update(self, o, l, r, idx, ch):
        if l == r:
            self.lc[o] = ch
            self.rc[o] = ch
            return
        m = (l + r) // 2
        if idx <= m:
            self._update(2 * o, l, m, idx, ch)
        else:
            self._update(2 * o + 1, m + 1, r, idx, ch)
        self._pull(o)


def longest_repeating(s, query_characters, query_indices):
    st = SegTree(s)
    res = []
    for ch, i in zip(query_characters, query_indices):
        st.update(i, ch)
        res.append(st.bs[1])
    return res


if __name__ == "__main__":
    assert longest_repeating("babacc", "bcb", [1, 3, 3]) == [3, 3, 4]
    assert longest_repeating("abyzz", "aa", [2, 1]) == [2, 3]
    assert longest_repeating("a", "b", [0]) == [1]
    print("longest_repeating_character: all tests passed")
