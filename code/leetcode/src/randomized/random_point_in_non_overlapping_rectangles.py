"""497. 非重叠矩形中的随机点（Random Point in Non-overlapping Rectangles）

题目：给定若干互不重叠的轴对齐矩形 rects[i] = [x1, y1, x2, y2]（含边界整数格点），
    实现 pick()：先按面积比例等概率选一个矩形，再在该矩形内等概率取一个整数点。
    等价地说，每个整数点被选中的概率正比于它所在矩形的面积。

思路（前缀和 + 二分 + 矩形内均匀）：
    矩形 i 包含的整数点个数是 (x2 - x1 + 1) × (y2 - y1 + 1)。把每个矩形的点数当作
    权重做前缀和，就得到它在「总点数」数轴上的区间。pick() 时在 [1, 总点数] 上均匀取
    target，二分（lower_bound）定位落在哪个矩形；再在该矩形内对 x、y 各自均匀取一个
    整数，即得等概率的整数点。

    为什么先矩形后坐标就够了？因为从总面积层面按比例选矩形、矩形内部再均匀取点，
    两者相乘后，每个点被选中的概率恰好是 1 / 总点数，与点的位置无关。

复杂度：初始化 O(n)；pick O(log n)；空间 O(n)。
"""

import bisect
import random


class Solution:
    def __init__(self, rects):
        self.rects = rects
        self.prefix = []
        total = 0
        for x1, y1, x2, y2 in rects:
            total += (x2 - x1 + 1) * (y2 - y1 + 1)
            self.prefix.append(total)

    def pick(self):
        target = random.randint(1, self.prefix[-1])
        i = bisect.bisect_left(self.prefix, target)
        x1, y1, x2, y2 = self.rects[i]
        return [random.randint(x1, x2), random.randint(y1, y2)]


if __name__ == "__main__":
    random.seed(0)
    rects = [[1, 1, 5, 5], [-2, -2, 0, 0]]
    s = Solution(rects)
    assert s.prefix == [25, 34]
    covered = set()
    for _ in range(20000):
        x, y = s.pick()
        assert any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in rects)
        covered.add((x, y))
    assert len(covered) == 34
    print("random_point_in_non_overlapping_rectangles: all tests passed")
