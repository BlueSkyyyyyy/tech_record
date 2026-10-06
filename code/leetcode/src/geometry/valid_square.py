"""593. 有效的正方形（Valid Square）

题目：给定四个点，判断它们能否组成一个正方形（四条边相等、四个角为直角）。

思路（只看六个两两距离）：
    四个点两两之间一共有 C(4,2) = 6 条线段：其中 4 条是正方形的边，
    2 条是对角线。把 6 个**距离的平方**排序后，正方形必须满足：
    - 前 4 个相等且大于 0（四条边等长，且不能退化成同一个点）；
    - 后 2 个相等，且等于边长的 2 倍（对角线平方 = 边平方 + 边平方）。
    用距离平方可以躲开开根号的浮点误差，全部是整数运算。

    为什么这套条件充分：四点里必定有一组满足「勾股 + 四条边相等」的
    排列，正是正方形的定义；反之正方形的六个距离必然长这样。

复杂度：时间 O(1)（常数个点，排序 6 个数），空间 O(1)。
"""


def valid_square(p1, p2, p3, p4):
    pts = [p1, p2, p3, p4]
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dists.append(dx * dx + dy * dy)
    dists.sort()
    side, diag = dists[0], dists[4]
    return (side > 0 and dists[0] == dists[1] == dists[2] == dists[3]
            and diag == dists[5] == 2 * side)


if __name__ == "__main__":
    assert valid_square([0, 0], [1, 1], [1, 0], [0, 1]) is True
    assert valid_square([0, 0], [1, 1], [1, 0], [0, 12]) is False
    assert valid_square([1, 0], [-1, 0], [0, 1], [0, -1]) is True
    assert valid_square([0, 0], [0, 0], [0, 0], [0, 0]) is False
    print("valid_square: all tests passed")
