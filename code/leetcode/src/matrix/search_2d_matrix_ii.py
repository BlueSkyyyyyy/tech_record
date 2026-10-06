"""240. 搜索二维矩阵 II（Search a 2D Matrix II）

题目：给定 m x n 矩阵，每一行从左到右升序，每一列从上到下升序。判断目标值
target 是否存在于矩阵中，要求尽量高效。

思路（从右上角出发的「阶梯查找」）：
    从矩阵右上角 (0, n-1) 出发，反复做如下比较：
      - 若当前值等于 target，找到了；
      - 若当前值大于 target，说明它所在的这一列中，它下方的数只会更大，不可能
        有 target，于是列下标左移一列；
      - 若当前值小于 target，说明它所在的这一行中，它左边的数只会更小，不可能
        有 target，于是行下标下移一行。
    每比较一次就排除一整行或一整列，最多走 m+n 步。

    为什么必须从右上角（或左下角）出发：右上角是这一行里最大的、同时是这一列里
    最小的，所以一次比较就能确定「往左」还是「往下」。若从左上角出发，它是行和
    列的最小值，两个方向的值都比它大，无法排除；从右下角同理。

复杂度：时间 O(m+n)（每步排除一行或一列），空间 O(1)。
"""


def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    i, j = 0, len(matrix[0]) - 1
    while i < len(matrix) and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False


if __name__ == "__main__":
    matrix = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    assert search_matrix(matrix, 5)
    assert search_matrix(matrix, 30)
    assert search_matrix(matrix, 1)
    assert not search_matrix(matrix, 20)
    assert not search_matrix(matrix, 0)
    assert not search_matrix(matrix, 31)
    assert search_matrix([[1]], 1)
    assert not search_matrix([[1]], 2)
    assert not search_matrix([], 1)
    print("search_matrix: all tests passed")
