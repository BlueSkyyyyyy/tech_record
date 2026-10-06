"""73. 矩阵置零（Set Matrix Zeroes）

题目：给定 m x n 矩阵 matrix，如果某个元素为 0，则把它所在的整行和整列都置为
0。要求原地修改。

思路（借首行、首列当标记位）：
    不能一遇到 0 就立刻把整行整列清零，那样会把后面还没读取的格子也污染掉。
    正确做法分三步：
      1) 用变量 col0 记下「第 0 列本身是否含有 0」；然后扫描除第 0 行、第 0 列
         以外的所有格子，若 matrix[i][j] == 0，就在首列/首行对应位置做标记：
         matrix[i][0] = 0、matrix[0][j] = 0。
      2) 再扫一遍这些格子，只要它所在行的标记（matrix[i][0]）或所在列的标记
         （matrix[0][j]）为 0，就把它置 0。
      3) 最后补上第 0 行和第 0 列：matrix[0][0] == 0 说明第 0 行要清零，
         col0 为真说明第 0 列要清零。

    为什么能省到 O(1) 空间：矩阵的首行、首列本身就可以充当「这一行/列要不要
    清零」的备忘录，代价只是要提前把它们自己的信息挪到两个变量里保存：
    第 0 行的信息用 matrix[0][0]，第 0 列的信息用 col0。
"""


def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    col0 = any(matrix[i][0] == 0 for i in range(m))
    for i in range(m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if matrix[0][0] == 0:
        for j in range(n):
            matrix[0][j] = 0
    if col0:
        for i in range(m):
            matrix[i][0] = 0


if __name__ == "__main__":
    m = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes(m)
    assert m == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]

    m = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    set_zeroes(m)
    assert m == [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]

    m = [[1, 2, 3], [4, 5, 6]]
    set_zeroes(m)
    assert m == [[1, 2, 3], [4, 5, 6]]

    m = [[1], [0]]
    set_zeroes(m)
    assert m == [[0], [0]]

    print("set_zeroes: all tests passed")
