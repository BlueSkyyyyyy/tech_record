"""223. 矩形面积（Rectangle Area）

题目：给出两个轴对齐矩形的左下角与右上角坐标（第一个矩形
(ax1, ay1, ax2, ay2)，第二个 (bx1, by1, bx2, by2)），返回两个矩形覆盖的
总面积（重叠部分只算一次）。

思路（容斥原理）：
    两个矩形面积之和减去重叠面积，就是并集面积。重叠区域本身还是一个轴对齐
    矩形，它的宽是「两矩形右边界较小值 − 左边界较大值」，高同理；两者都
    取 max(0, …)，因为没有重叠时这个差值会为负，直接当作 0。

    为什么用 min(max) 就能求交：一维区间 [a1,a2] 与 [b1,b2] 的交是
    [max(a1,b1), min(a2,b2)]，长度是端点之差；二维矩形分别对 x、y 求交再相乘
    即可。这正是二维前缀和容斥（第 04 篇）里「交集」的另一种用法。

复杂度：时间 O(1)，空间 O(1)。
"""


def compute_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return area_a + area_b - width * height


if __name__ == "__main__":
    assert compute_area(-3, 0, 3, 4, 0, -1, 9, 2) == 45
    assert compute_area(0, 0, 0, 0, 0, 0, 0, 0) == 0
    assert compute_area(0, 0, 2, 2, 1, 1, 3, 3) == 7
    assert compute_area(0, 0, 1, 1, 2, 2, 3, 3) == 2
    print("rectangle_area: all tests passed")
