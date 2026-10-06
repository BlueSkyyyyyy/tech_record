"""867. 转置矩阵（Transpose Matrix）

题目：给定一个 m x n 的矩阵 matrix，返回它的转置矩阵（n x m），即把原矩阵的
行变成列、列变成行，元素满足 result[j][i] == matrix[i][j]。

思路（下标交换 + zip）：
    转置的本质就是交换两个下标。Python 里最直接的写法是 zip(*matrix)：把每一行
    当作一个参数摊开来，zip 会把各行的第 0 个、第 1 个、……元素分别打包成新的
    行，正好就是转置后的列。zip 返回的是元组，需要再转成 list。

    为什么不必双重循环：双重循环（result[j][i] = matrix[i][j]）完全正确，而且
    在 C++ 里就是这么写的；Python 的 zip 只是同一件事更短的表达。理解上要记住
    「转置 = 交换行列下标」这一条，语言怎么写是次要的。

复杂度：时间 O(m*n)（每个元素搬一次），空间 O(m*n)（返回结果本身）。
"""


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


if __name__ == "__main__":
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
    assert transpose([[7]]) == [[7]]
    assert transpose([[1, 2]]) == [[1], [2]]
    assert transpose([[1], [2]]) == [[1, 2]]
    print("transpose: all tests passed")
