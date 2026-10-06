"""1409. 查询带键的排列（Queries on a Permutation With Key）

题目：给定 m 和查询数组 queries。初始排列 P = [1, 2, ..., m]。对每个查询 q：
    找到 q 在 P 中的位置（下标从 0 开始），把它记入答案，然后把 q 移到 P 的开头。
    返回所有查询的位置组成的数组。

思路（树状数组维护「前边还有多少个元素」）：
    朴素做法每次找位置、移动元素都是 O(m)，总 O(m·n)。用树状数组可以把「找位置」
    和「移到开头」都做到 O(log(m+n))。

    技巧是**预留空间**：开一个大小 m + n 的数组（下标 1..m+n 给树状数组用），
    把初始的 1..m 放在靠后的 n+1..n+m 位置，前面 n 个位置空出来给「移到开头」用。
    树状数组里每个「已占用」位置记为 1，那么元素 x 当前的 0 基下标 = 它所在位置
    左边已占用的元素个数 = prefix(pos[x] - 1)。

    移动 x 到开头：把 pos[x] 处减 1，在「前面的空槽」下一个可用位置加 1，并更新
    pos[x]。因为每次移动用一个空槽，n 次查询最多用掉 n 个空槽，正好。

    为什么不直接维护下标：元素一旦移动，其它元素下标全变。树状数组把「位置」当成
    槽位来管理，元素只是占用某个槽，槽的占用前缀和就是它的动态下标。

复杂度：时间 O((m + n) log(m + n))，空间 O(m + n)。
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


def process_queries(queries, m):
    n = len(queries)
    bit = Fenwick(m + n)

    pos = [0] * (m + 1)
    for v in range(1, m + 1):
        pos[v] = n + v
        bit.add(pos[v], 1)

    ans = []
    next_pos = n
    for q in queries:
        p = pos[q]
        ans.append(bit.prefix(p - 1))
        bit.add(p, -1)
        bit.add(next_pos, 1)
        pos[q] = next_pos
        next_pos -= 1
    return ans


if __name__ == "__main__":
    assert process_queries([3, 1, 2, 1], 5) == [2, 1, 2, 1]
    assert process_queries([4, 1, 2, 2], 4) == [3, 1, 2, 0]
    assert process_queries([7, 5, 5, 8, 3], 8) == [6, 5, 0, 7, 5]
    assert process_queries([1], 1) == [0]

    print("process_queries: all tests passed")
