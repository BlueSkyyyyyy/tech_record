"""48. 旋转图像（Rotate Image）

题目：给定一个 n x n 的二维矩阵 matrix，原地（不新建矩阵）把它顺时针旋转 90 度。

思路（转置 + 水平翻转）：
    「顺时针旋转 90 度」可以拆成两个已经会的动作：
      1) 沿主对角线转置：交换 matrix[i][j] 与 matrix[j][i]；
      2) 再对每一行做水平翻转（左右颠倒）。
    两步复合的结果恰好就是顺时针旋转 90 度。

    为什么恰好是这两步：设原坐标是 (i, j)。转置后到 (j, i)，再水平翻转把列
    坐标变成 n-1-i，得到 (j, n-1-i)——这正是顺时针 90 度的坐标变换。分解成两步
    的好处是，每一步都只是「交换两个元素」，下标不容易写错。

    另一种等价写法是「一圈一圈做四元交换」（上→右→下→左），不用额外空间也
    只交换一次，但边界处理更绕；转置 + 翻转更直观，建议先掌握这一版。
"""


def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()


if __name__ == "__main__":
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rotate(m)
    assert m == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    m = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]
    rotate(m)
    assert m == [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]]

    m = [[1]]
    rotate(m)
    assert m == [[1]]

    m = [[1, 2], [3, 4]]
    rotate(m)
    assert m == [[3, 1], [4, 2]]

    print("rotate: all tests passed")
