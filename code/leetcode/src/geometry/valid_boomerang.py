"""1037. 有效的回旋镖（Valid Boomerang）

题目：给定平面上的三个点 points = [p1, p2, p3]，判断它们是否两两不同且
不共线（即能构成一个非退化的三角形）。是则返回 true。

思路（向量叉积判共线）：
    以 p1 为起点作两个向量 u = p2 - p1，v = p3 - p1。叉积
    u × v = u.x * v.y - u.y * v.x。
    - 叉积为 0 表示两向量平行，即三点共线；
    - 叉积非 0 表示三点不共线，构成有效三角形。
    用叉积而不是斜率，因为斜率要处理「垂直于 x 轴」的特殊情况，叉积则统一
    用一次乘减搞定，既短又稳。

    题目保证三个点两两不同，所以只需判叉积非零。

复杂度：时间 O(1)，空间 O(1)。
"""


def is_boomerang(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) != 0


if __name__ == "__main__":
    assert is_boomerang([[1, 1], [2, 3], [3, 2]]) is True
    assert is_boomerang([[1, 1], [2, 2], [3, 3]]) is False
    assert is_boomerang([[0, 0], [0, 1], [1, 0]]) is True
    assert is_boomerang([[0, 0], [1, 0], [2, 0]]) is False
    print("valid_boomerang: all tests passed")
