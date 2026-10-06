"""528. 按权重随机选择（Random Pick with Weight）

题目：给定正整数数组 w，w[i] 表示下标 i 的权重。实现 pickIndex()，按下标 i 被选中的
    概率为 w[i] / sum(w) 的分布返回一个下标。

思路（前缀和 + 二分查找）：
    把权重摊到一条数轴上：下标 0 占据 [0, w[0])，下标 1 占据 [w[0], w[0]+w[1])，以此类推，
    整条数轴总长 sum(w)。只要在总长上等概率地取一个整数 target，它落在哪一段里，
    就返回那一段对应的下标——这样区间越长（权重越大）命中概率越高，正好是加权抽样。

    前缀和 prefix[i] = w[0] + … + w[i] 记录每一段的右端点（1-based，方便取值）。
    target 落在下标 i 的区间，等价于「第一个满足 prefix[i] >= target 的 i」，
    用 lower_bound（bisect_left）二分即可，不必顺序扫描。

复杂度：初始化 O(n)；pickIndex O(log n)；空间 O(n)。
"""

import bisect
import random


class Solution:
    def __init__(self, w):
        self.prefix = []
        total = 0
        for x in w:
            total += x
            self.prefix.append(total)

    def pickIndex(self):
        target = random.randint(1, self.prefix[-1])
        return bisect.bisect_left(self.prefix, target)


if __name__ == "__main__":
    random.seed(0)
    s = Solution([1, 2, 3])
    assert s.prefix == [1, 3, 6]
    counts = [0, 0, 0]
    for _ in range(60000):
        i = s.pickIndex()
        assert 0 <= i < 3
        counts[i] += 1
    assert counts[0] < counts[1] < counts[2]
    assert abs(counts[0] / 60000 - 1 / 6) < 0.02
    assert abs(counts[2] / 60000 - 3 / 6) < 0.02
    print("random_pick_with_weight: all tests passed")
