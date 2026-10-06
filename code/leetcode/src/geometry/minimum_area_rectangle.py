"""939. 最小面积矩形（Minimum Area Rectangle）

题目：给定平面上一些互不相同的整数坐标点，求由其中 4 个点构成的**边平行于
坐标轴**的矩形的最小面积；不存在这样的矩形返回 0。

思路（按竖边分组，记住每对 y 上次出现的横坐标）：
    轴对齐矩形由两条竖边确定：左竖边在 x = x1 上、横跨 (y1, y2)，右竖边在
    x = x2 上、同样横跨 (y1, y2)，且四个顶点都在点集里。
    于是把点按 x 分组，每组内的 y 值两两组成一条「竖直区间」(y1, y2)。
    从左到右扫描每个 x：
    - 若这个 (y1, y2) 之前出现过，出现的位置 last[(y1, y2)] 就是左竖边的 x，
      此时矩形面积 = (x - last[(y1, y2)]) * (y2 - y1)，更新最小值；
    - 然后把这个 (y1, y2) 的「最近出现 x」更新为当前 x。
    因为同一组内 y 唯一（点互不相同），枚举组内所有 y 对不会重复。

    为什么记「最近一次」就够：面积 = 宽 × 高，高固定时宽越小面积越小，
    所以对每个 (y1, y2) 只需保留最近的左边界。

复杂度：最坏每个 x 组内 y 对数为 O(m^2)，总体 O(n^2)（哈希操作为均摊
O(1)）；空间 O(n^2) 存放 (y1, y2) 键（n ≤ 500 时可行）。
"""

from collections import defaultdict


def min_area_rect(points):
    by_x = defaultdict(list)
    for x, y in points:
        by_x[x].append(y)

    last = {}
    best = float("inf")
    for x in sorted(by_x):
        ys = sorted(by_x[x])
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                key = (ys[i], ys[j])
                if key in last:
                    area = (x - last[key]) * (ys[j] - ys[i])
                    if area < best:
                        best = area
                last[key] = x
    return 0 if best == float("inf") else best


if __name__ == "__main__":
    assert min_area_rect([[1, 1], [1, 3], [3, 1], [3, 3], [2, 2]]) == 4
    assert min_area_rect([[1, 1], [1, 3], [3, 1], [3, 3], [4, 1], [4, 3]]) == 2
    assert min_area_rect([[1, 1], [2, 2], [3, 3]]) == 0
    print("minimum_area_rectangle: all tests passed")
