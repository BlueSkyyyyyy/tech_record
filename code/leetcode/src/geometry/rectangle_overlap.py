"""836. 矩形重叠（Rectangle Overlap）

题目：给定两个轴对齐矩形 rec1 与 rec2，每个用 [x1, y1, x2, y2] 表示
（(x1,y1) 左下、(x2,y2) 右上）。若它们重叠（交集面积大于 0）返回 true。

思路（两个方向都重叠才算重叠）：
    矩形重叠 ⟺ 它们在 x 轴上的投影相交，且在 y 轴上的投影也相交。
    两个区间相交的充要条件是「左端点的最大值 < 右端点的最小值」，即
    max(x1) < min(x2)。这里刻意用严格小于：只接触一条边或一个点不算重叠。

    等价的反面写法是「一个矩形完全在另一个的左边 / 右边 / 上边 / 下边」，
    但直接写相交更短、更不容易漏条件。

复杂度：时间 O(1)，空间 O(1)。
"""


def is_rectangle_overlap(rec1, rec2):
    x_overlap = min(rec1[2], rec2[2]) > max(rec1[0], rec2[0])
    y_overlap = min(rec1[3], rec2[3]) > max(rec1[1], rec2[1])
    return x_overlap and y_overlap


if __name__ == "__main__":
    assert is_rectangle_overlap([0, 0, 2, 2], [1, 1, 3, 3]) is True
    assert is_rectangle_overlap([0, 0, 1, 1], [1, 0, 2, 1]) is False
    assert is_rectangle_overlap([0, 0, 1, 1], [2, 2, 3, 3]) is False
    assert is_rectangle_overlap([0, 0, 3, 3], [1, 1, 2, 2]) is True
    assert is_rectangle_overlap([7, 8, 13, 15], [10, 8, 12, 20]) is True
    print("rectangle_overlap: all tests passed")
