"""LCR 170. 交易逆序对总数（原剑指 Offer 51. 数组中的逆序对）

题目：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个
    逆序对。求数组中逆序对的总数。

思路（值域树状数组）：
    「数前面有多少个比当前数大」这类问题，可以边扫描边用树状数组维护已经出现过的
    数字的**值域计数**：
      - 把出现的数字 x 在「值域坐标」rank(x) 上 +1；
      - 扫到 x 时，比 x 大的已出现个数 = 已插入总数 - prefix(rank(x))。
    把每个位置贡献的「前面比它大」累加，就是逆序对总数。

    这里的关键是**离散化**：数字的大小可能很大（甚至有负数），但真正用到的只有
    「大小排名」。把所有出现过的值排序去重，用二分或字典映射到 1..m 的排名，
    树状数组只需开 m 大小。

    同类问题有三种等价问法，扫的方向和「大/小」对应着变：
      - 从左往右 + 「前面比我大」= 逆序对；
      - 从右往左 + 「后面比我小」= 逆序对（本例 315 的做法）；
      - 从左往右 + 「前面比我小」= 顺序对。

复杂度：时间 O(n log n)（每次查询 / 插入各 O(log n)），空间 O(n)。
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


def reverse_pairs_count(record):
    n = len(record)
    if n == 0:
        return 0
    sorted_vals = sorted(set(record))
    rank = {v: i + 1 for i, v in enumerate(sorted_vals)}
    bit = Fenwick(len(sorted_vals))

    ans = 0
    processed = 0
    for x in record:
        r = rank[x]
        ans += processed - bit.prefix(r)
        bit.add(r, 1)
        processed += 1
    return ans


if __name__ == "__main__":
    assert reverse_pairs_count([7, 5, 6, 4]) == 5
    assert reverse_pairs_count([1, 2, 3, 4]) == 0
    assert reverse_pairs_count([4, 3, 2, 1]) == 6
    assert reverse_pairs_count([1]) == 0
    assert reverse_pairs_count([]) == 0
    assert reverse_pairs_count([-1, -2, 0, -1]) == 2
    assert reverse_pairs_count([2, 2, 2]) == 0

    print("reverse_pairs_count: all tests passed")
