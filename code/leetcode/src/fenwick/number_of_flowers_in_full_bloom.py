"""2251. 花期内花的数目（Number of Flowers in Full Bloom）

题目：flowers[i] = [start, end] 表示第 i 朵花从 start 开到 end（含两端）。
people[i] 表示第 i 个人到达的时刻。返回每个人到达时「正在开花」的花的数目。

思路（区间加 → 差分 + 树状数组单点查询）：
    一个人看到的不是某一朵花，而是「有多少个区间盖住了这个时刻」。把每朵花看成
    时间轴上的一次区间加 +1：[start, end] 整体加一。用差分把它拆成两次单点修改：
        diff[start] += 1,  diff[end + 1] -= 1
    某个时刻 t 的答案就是 diff 的前缀和。时刻可以到 1e9，所以先把所有 start、
    end + 1 与 people 的坐标离散化，再用树状数组维护这台「差分前缀和」：
        add(rank(start), +1); add(rank(end + 1), -1);
        answer(t) = prefix(rank(t))

    为什么用树状数组而不是普通差分数组：坐标离散化后，树状数组仍然是 O(log n) 的
    单点修改 + 前缀查询，和前面「值域计数」用的是同一份模板；代价是常数比裸差分数组大。
    如果所有查询一次性给全、且坐标范围不大，排序 + 差分数组的前缀和 O(n + m) 更快。

复杂度：预处理 O(n log n)，每次查询 O(log m)，空间 O(n + m)。
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
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total


def full_bloom_flowers(flowers, people):
    coords = set(people)
    for start, end in flowers:
        coords.add(start)
        coords.add(end + 1)
    xs = sorted(coords)
    rank = {x: i + 1 for i, x in enumerate(xs)}

    bit = Fenwick(len(xs))
    for start, end in flowers:
        bit.add(rank[start], 1)
        bit.add(rank[end + 1], -1)

    return [bit.prefix(rank[t]) for t in people]


if __name__ == "__main__":
    assert full_bloom_flowers(
        [[1, 6], [3, 7], [9, 12], [4, 13]], [2, 3, 7, 11]
    ) == [1, 2, 2, 2]
    assert full_bloom_flowers([[1, 10], [3, 3]], [3, 3, 2]) == [2, 2, 1]
    assert full_bloom_flowers([[1, 1]], [1, 2]) == [1, 0]
    assert full_bloom_flowers([[1, 5]], [0, 6]) == [0, 0]
    print("number_of_flowers_in_full_bloom: all tests passed")
