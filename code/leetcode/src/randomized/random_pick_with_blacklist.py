"""710. 黑名单中的随机数（Random Pick with Blacklist）

题目：给定整数 n 和黑名单 blacklist（数字互不相同、均在 [0, n) 内），实现 pick()：
    等概率返回 [0, n) 中一个不在黑名单里的整数。

思路（把黑名单「换」到尾部）：
    设 m = len(blacklist)，可用的数共 size = n - m 个。我们只在区间 [0, size) 上等概率
    取一个下标；但如果这个下标本身是黑名单，就要把它映射到一个「尾部」的可用数
    （下标在 [size, n) 里）。

    建立映射：从 n-1 往下，跳过所有黑名单数字，找到最大的可用数 last；对每个落在
    [0, size) 的黑名单 b，令 mapping[b] = last，再让 last 继续往下找。
    落在 [size, n) 的黑名单不用映射——我们本来就不会去尾部取值。
    pick() 时取 idx ∈ [0, size)，命中 mapping 就返回映射值，否则返回 idx 本身。

    这样一来，[0, size) 里每个下标（无论是黑名单被映射过去的，还是原本就可用）都
    等概率且唯一地对应一个可用数，整体就是均匀分布。

复杂度：初始化 O(m)；pick O(1)（哈希表）；空间 O(m)。
"""

import random


class Solution:
    def __init__(self, n, blacklist):
        m = len(blacklist)
        self.size = n - m
        blocked = set(blacklist)
        self.mapping = {}
        last = n - 1
        for b in blacklist:
            if b < self.size:
                while last in blocked:
                    last -= 1
                self.mapping[b] = last
                last -= 1

    def pick(self):
        idx = random.randint(0, self.size - 1)
        return self.mapping.get(idx, idx)


if __name__ == "__main__":
    random.seed(0)
    s = Solution(7, [2, 3, 5])
    assert s.size == 4
    blocked = {2, 3, 5}
    seen = set()
    for _ in range(4000):
        x = s.pick()
        assert 0 <= x < 7 and x not in blocked
        seen.add(x)
    assert seen == {0, 1, 4, 6}

    s2 = Solution(5, [3, 4])
    assert {s2.pick() for _ in range(2000)} == {0, 1, 2}
    print("random_pick_with_blacklist: all tests passed")
