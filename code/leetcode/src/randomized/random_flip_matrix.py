"""519. 随机翻转矩阵（Random Flip Matrix）

题目：m×n 的矩阵初始全为 0。实现：
    flip()：等概率随机选一个当前值为 0 的格子，把它置 1，返回 [row, col]；
    reset()：把矩阵恢复成全 0。

思路（一维编号 + 「末尾填坑」映射）：
    把矩阵拉平成一维编号 0…total-1（total = m×n）。维护「还剩多少个 0」total，
    以及一个哈希表 mapping 记录「某些编号实际代表的格子」。

    flip：先 total -= 1（最后一个未选编号是 total），在 [0, total] 里等概率取 idx，
    真正要翻转的编号是 chosen = mapping.get(idx, idx)；随后用「当前末尾编号 total」
    去补 idx 的坑：mapping[idx] = mapping.get(total, total)。
    这样编号 total 被搬到了 idx 位置，下次若又抽到 idx，会得到 total 原来代表的格子，
    保证每个 0 格子恰好被等概率选中一次。
    reset：把 total 和 mapping 复位即可，不需要真的清矩阵。

复杂度：初始化 O(1)；flip / reset 平均 O(1)；空间 O(翻转过的不重复格子数)。
"""

import random


class Solution:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.total = m * n
        self.mapping = {}

    def flip(self):
        self.total -= 1
        idx = random.randint(0, self.total)
        chosen = self.mapping.get(idx, idx)
        self.mapping[idx] = self.mapping.get(self.total, self.total)
        return [chosen // self.n, chosen % self.n]

    def reset(self):
        self.total = self.m * self.n
        self.mapping = {}


if __name__ == "__main__":
    random.seed(0)
    s = Solution(3, 4)
    got = set()
    for _ in range(12):
        r, c = s.flip()
        assert 0 <= r < 3 and 0 <= c < 4
        got.add((r, c))
    assert len(got) == 12
    s.reset()
    got2 = {tuple(s.flip()) for _ in range(12)}
    assert len(got2) == 12
    print("random_flip_matrix: all tests passed")
