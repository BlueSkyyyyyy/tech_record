"""963. 最小面积矩形 II（Minimum Area Rectangle II）

题目：给定平面上一些点，求由其中 4 个点构成的**任意方向**矩形的最小面积；
不存在返回 0。

思路（矩形的对角线特征 + 分组枚举）：
    一个四边形是矩形，当且仅当它的两条对角线**互相平分且长度相等**。
    所以两条对角线的中点相同、长度相同。反过来，对于同一个中点、同一长度的
    两组点对，它们拼起来一定是一个矩形（两条相等的对角线互相平分）。

    做法：枚举所有点对 (i, j) 作为一条候选对角线，把它按
    「中点 (x1+x2, y1+y2) 的坐标和 + 对角线长度平方 (dx^2+dy^2)」分组。
    为了避开小数，中点的两倍坐标 (x1+x2, y1+y2) 与长度平方都用整数表示。
    同一个组里的任意两条对角线都能拼成矩形，面积 =
    |两条对角线向量的叉积| / 2，枚举组内两两组合取最小。

复杂度：枚举点对 O(n^2)，组内两两组合最坏 O(n^2)，总体 O(n^2)（n ≤ 50）；
空间 O(n^2)。
"""

from collections import defaultdict


def min_area_free_rect(points):
    n = len(points)
    groups = defaultdict(list)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            mid = (x1 + x2, y1 + y2)
            dist2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            groups[(mid, dist2)].append((i, j))

    best = float("inf")
    for pairs in groups.values():
        k = len(pairs)
        for a in range(k):
            i, j = pairs[a]
            d1x = points[j][0] - points[i][0]
            d1y = points[j][1] - points[i][1]
            for b in range(a + 1, k):
                p, q = pairs[b]
                d2x = points[q][0] - points[p][0]
                d2y = points[q][1] - points[p][1]
                area = abs(d1x * d2y - d1y * d2x)
                if area < best:
                    best = area
    return 0.0 if best == float("inf") else best / 2


if __name__ == "__main__":
    assert min_area_free_rect([[0, 1], [2, 1], [1, 1], [1, 0], [2, 0]]) == 1.0
    assert min_area_free_rect([[1, 2], [2, 1], [1, 0], [0, 1]]) == 2.0
    assert min_area_free_rect([[0, 0], [1, 1], [2, 2]]) == 0.0
    assert min_area_free_rect([[0, 0], [1, 1], [1, 0], [0, 1]]) == 1.0
    print("minimum_area_rectangle_ii: all tests passed")
