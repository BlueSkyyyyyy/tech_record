"""850. 矩形面积 II（Rectangle Area II）

题目：求一组轴对齐矩形覆盖的总面积（重叠只算一次），结果对 1e9 + 7 取模。

思路（按 x 离散化 + 逐条竖带求并集）：
    把所有矩形的左右 x 坐标收集起来排序去重，得到若干个「竖带」
    [xs[i], xs[i+1])。在一条竖带内部，是否被某个矩形覆盖不会变化
    （因为带内没有任何矩形的边界），所以：
    - 找出所有「横向完全盖住这条带」的矩形（`x1 <= xa 且 xb <= x2`）；
    - 把这些矩形的 y 区间求并集，得到这条带被覆盖的总高度；
    - 面积贡献 = 带宽 × 覆盖高度。
    各条带互不重叠，直接相加即可。y 区间求并集：按左端点排序后一次扫描合并。

复杂度：设矩形数 n，x 坐标去重后至多 2n 条带；每条带扫描 n 个矩形并排序，
       总体 O(n^2 log n)，空间 O(n)。（n <= 200 时足够。）
"""


def rectangle_area(rectangles):
    MOD = 10**9 + 7
    xs = sorted({x for x1, _, x2, _ in rectangles for x in (x1, x2)})
    area = 0
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        spans = []
        for x1, y1, x2, y2 in rectangles:
            if x1 <= xa and xb <= x2:
                spans.append((y1, y2))
        spans.sort()

        covered = 0
        cur_lo = cur_hi = None
        for y1, y2 in spans:
            if cur_hi is None:
                cur_lo, cur_hi = y1, y2
            elif y1 > cur_hi:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = y1, y2
            else:
                cur_hi = max(cur_hi, y2)
        if cur_hi is not None:
            covered += cur_hi - cur_lo

        area = (area + (xb - xa) * covered) % MOD
    return area


if __name__ == "__main__":
    assert rectangle_area([[0, 0, 2, 2], [1, 0, 2, 3], [1, 0, 3, 1]]) == 6
    assert rectangle_area([[0, 0, 1000000000, 1000000000]]) == 49
    assert rectangle_area([[0, 0, 1, 1], [2, 2, 3, 3]]) == 2
    print("rectangle_area_ii: all tests passed")
