"""478. 在圆内随机生成点（Generate Random Point in a Circle）

题目：给定半径 radius 和圆心 (x_center, y_center)，等概率返回圆内（含边界）的一个
    随机点。

思路（拒绝采样）：
    先画一个以圆心为中心、边长 2R 的外接正方形，在正方形里均匀取点
    （x、y 各自在 [-1, 1] 上均匀，再放大 R 倍平移）。若 x² + y² ≤ 1 就接受，
    否则重抽。正方形内的均匀分布被「裁剪」到圆内后，仍是圆内的均匀分布。
    接受概率 = πR² / (4R²) = π/4 ≈ 0.785，期望重抽约 1.27 次。

    若改用极坐标，半径必须取 sqrt(U)·R（U 均匀），才能保证面积均匀；直接取 R·U 会让
    点向圆心聚集。拒绝采样的好处是无需推导、不会踩这个坑。

复杂度：期望时间 O(1)，空间 O(1)。
"""

import random


class Solution:
    def __init__(self, radius, x_center, y_center):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self):
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x * x + y * y <= 1:
                return [self.x_center + x * self.radius,
                        self.y_center + y * self.radius]


if __name__ == "__main__":
    random.seed(0)
    s = Solution(1.0, 0.0, 0.0)
    for _ in range(5000):
        x, y = s.randPoint()
        assert x * x + y * y <= 1.0 + 1e-9
    s2 = Solution(2.0, 1.0, -1.0)
    for _ in range(5000):
        x, y = s2.randPoint()
        assert (x - 1) ** 2 + (y + 1) ** 2 <= 4.0 + 1e-9
    print("generate_random_point_in_a_circle: all tests passed")
