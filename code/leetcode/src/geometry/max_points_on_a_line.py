"""149. 直线上最多的点数（Max Points on a Line）

题目：给定平面上一组互不相同的点，求最多有多少个点在同一条直线上。

思路（枚举一个点当「基准」，把方向归一化后哈希计数）：
    一条直线可以由「过某点 + 一个方向」唯一确定。于是枚举每个点 i 作为
    基准，再枚举它后面的点 j，把向量 (dx, dy) = points[j] - points[i]
    化到最简形式当作这条线的「方向指纹」，用哈希表统计各方向出现次数。
    过 i 且沿某方向的直线上，点数 = 该方向的计数 + 1（加上 i 自己）。

    方向归一化的三步：
    1. 用 gcd 约分：dx, dy 同时除以 gcd(|dx|, |dy|)，得到互质的整数对；
    2. 统一符号：规定方向 (dx, dy) 满足 dx > 0，或 dx == 0 时 dy > 0。
       这样 (1, 2) 与 (-1, -2) 会被归成同一个方向，不会重复计数；
    3. 用 (dx, dy) 元组当哈希键。

    关键点：**全程用整数**，不写斜率 dy/dx 浮点数，避免 1/3 与 2/6 因
    浮点误差被当成不同方向；gcd 约分让它们变成同一个键。

    复杂度：枚举基准 O(n)，每个基准再扫一遍 O(n)，每对点求一次 gcd
    O(log C)，总体 O(n^2 log C)，n ≤ 300 足够快；空间 O(n)（哈希表）。
"""

from collections import defaultdict
from math import gcd


def max_points(points):
    n = len(points)
    if n <= 2:
        return n
    best = 0
    for i in range(n):
        xi, yi = points[i]
        directions = defaultdict(int)
        for j in range(i + 1, n):
            dx = points[j][0] - xi
            dy = points[j][1] - yi
            g = gcd(abs(dx), abs(dy)) or 1
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            directions[(dx, dy)] += 1
        if directions:
            best = max(best, 1 + max(directions.values()))
    return best


if __name__ == "__main__":
    assert max_points([[1, 1], [2, 2], [3, 3]]) == 3
    assert max_points([[1, 1], [3, 2], [5, 3], [4, 1], [2, 3], [1, 4]]) == 4
    assert max_points([[0, 0]]) == 1
    assert max_points([[0, 0], [1, 1]]) == 2
    assert max_points([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1]]) == 4
    print("max_points_on_a_line: all tests passed")
