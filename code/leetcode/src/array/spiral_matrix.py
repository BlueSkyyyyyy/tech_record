"""54. 螺旋矩阵（Spiral Matrix）

题目：给定一个 m x n 的矩阵，按顺时针螺旋顺序返回矩阵中的所有元素。

思路（四条边界模拟）：
    用 top、bottom、left、right 四条边界框住「尚未访问的子矩形」，然后按
    右、下、左、上四个方向依次走一条边，每走完一条边就把对应边界向内收一格：
        右：从 left 到 right，走完后 top += 1；
        下：从 top 到 bottom，走完后 right -= 1；
        左：从 right 到 left，走完后 bottom -= 1；
        上：从 bottom 到 top，走完后 left += 1。
    为什么收完右/下之后要判断 top <= bottom、left <= right：当剩余区域只剩
    一行或一列时，如果还照直走「左」和「上」，会把同一行/列重复访问一遍。
    加上这两个判断，就只在确实还有剩余行/列时才继续。

复杂度：时间 O(m * n)，每个元素恰好访问一次；空间 O(1)（不计输出）。
"""


def spiral_order(matrix):
    if not matrix or not matrix[0]:
        return []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    res = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            res.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            res.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                res.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1
    return res


if __name__ == "__main__":
    assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [
        1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7
    ]
    assert spiral_order([[1]]) == [1]
    assert spiral_order([[1, 2], [3, 4]]) == [1, 2, 4, 3]
    assert spiral_order([]) == []
    assert spiral_order([[]]) == []
    print("spiral_matrix: all tests passed")
