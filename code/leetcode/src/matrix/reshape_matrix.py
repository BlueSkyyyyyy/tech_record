"""566. 重塑矩阵（Reshape the Matrix）

题目：给定 m x n 矩阵 mat 和目标形状 r x c。若元素总数相同（m*n == r*c），
把 mat 按「逐行展开」的顺序重新排成 r 行 c 列；否则原样返回。

思路（用一维下标当桥梁）：
    重塑前后元素的「行优先顺序」完全一致。于是把原矩阵的每个元素看成数组里的
    第 k 个元素（k 从 0 数到 m*n-1），转换公式是：

        原矩阵位置：(k // n, k % n)
        新矩阵位置：(k // c, k % c)

    遍历 k，从原矩阵按第一个公式取出，再按第二个公式放进新矩阵即可。

    为什么要先判元素总数：总数对不上时目标形状根本不存在，题目要求原样返回，
    这是本题最容易漏掉的边界。另外注意「按行优先」这个前提，不能按列展开。
"""


def matrix_reshape(mat, r, c):
    m, n = len(mat), len(mat[0])
    if m * n != r * c:
        return mat
    res = [[0] * c for _ in range(r)]
    for k in range(m * n):
        res[k // c][k % c] = mat[k // n][k % n]
    return res


if __name__ == "__main__":
    assert matrix_reshape([[1, 2], [3, 4]], 1, 4) == [[1, 2, 3, 4]]
    assert matrix_reshape([[1, 2], [3, 4]], 2, 4) == [[1, 2], [3, 4]]
    assert matrix_reshape([[1, 2], [3, 4]], 4, 1) == [[1], [2], [3], [4]]
    assert matrix_reshape([[1, 2, 3, 4]], 2, 2) == [[1, 2], [3, 4]]
    print("matrix_reshape: all tests passed")
