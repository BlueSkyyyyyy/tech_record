"""59. 螺旋矩阵 II（Spiral Matrix II）

题目：给定正整数 n，生成一个 n x n 的方阵，按顺时针方向从外到内螺旋填充
1 到 n*n。

思路（四条边界 + 逐步收缩）：
    维护四个边界 top、bottom、left、right。每一轮依次做四件事：
      - 从左到右填 top 这一行，填完 top += 1；
      - 从上到下填 right 这一列，填完 right -= 1；
      - 若还有剩余行，从右到左填 bottom 这一行，填完 bottom -= 1；
      - 若还有剩余列，从下到上填 left 这一列，填完 left += 1。
    每填一个方向就把对应的边界向内收一格，下一轮自然从更里面的一圈开始。

    为什么中间两个方向要再判断一次 top <= bottom / left <= right：当 n 为奇数
    时，最内圈可能只剩单独一行（或一列）。如果不判断就继续填，会把刚刚填过的
    位置反向覆盖一遍。加上这两个判断，循环就严格在还有空间时才走。

    这道题是「螺旋矩阵」（54 题）的生成版本：54 是把已有矩阵按螺旋顺序读出来，
    思路同样是这组边界，只是把「写值」换成「读值」。
"""


def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
    return matrix


if __name__ == "__main__":
    assert generate_matrix(1) == [[1]]
    assert generate_matrix(2) == [[1, 2], [4, 3]]
    assert generate_matrix(3) == [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
    assert generate_matrix(4) == [
        [1, 2, 3, 4],
        [12, 13, 14, 5],
        [11, 16, 15, 6],
        [10, 9, 8, 7],
    ]
    print("generate_matrix: all tests passed")
