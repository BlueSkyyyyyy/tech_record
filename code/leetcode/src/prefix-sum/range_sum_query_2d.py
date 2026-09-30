"""304. 二维区域和检索 - 矩阵不可变（Range Sum Query 2D - Immutable）

题目：给定二维矩阵 matrix，支持多次查询某个子矩阵的元素和。子矩阵由左上角 (row1, col1)
      和右下角 (row2, col2) 确定。矩阵在整个过程中不会被修改。
      例如 matrix = [[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]，
      sum_region(2,1,4,3) = 8，sum_region(1,1,2,2) = 11。

思路：把一维前缀和推广到二维。设 prefix[i][j] 表示「以 (0,0) 为左上角、
      (i-1, j-1) 为右下角」的子矩阵元素和；prefix 比 matrix 多一行一列，
      多出来的第 0 行/列全为 0，作用和一维里的 prefix[0] = 0 一样，用来兜住 0 下标。
      建表用容斥：
          prefix[i+1][j+1] = matrix[i][j] + prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
      含义是「当前格子 + 上方矩形 + 左方矩形 - 左上角重复加的那块」。
      查询也同理，子矩阵 (row1, col1)-(row2, col2) 的和为：
          prefix[row2+1][col2+1] - prefix[row1][col2+1] - prefix[row2+1][col1] + prefix[row1][col1]
      即「右下角大矩形」减去「上边一条」再减去「左边一条」，加回被多减的左上角小矩形。

复杂度：预处理时间 O(m*n)、空间 O(m*n)；每次查询 O(1)。
"""


class NumMatrix:
    def __init__(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (
                    matrix[i][j]
                    + self.prefix[i][j + 1]
                    + self.prefix[i + 1][j]
                    - self.prefix[i][j]
                )

    def sum_region(self, row1, col1, row2, col2):
        return (
            self.prefix[row2 + 1][col2 + 1]
            - self.prefix[row1][col2 + 1]
            - self.prefix[row2 + 1][col1]
            + self.prefix[row1][col1]
        )


if __name__ == "__main__":
    matrix = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5],
    ]
    nm = NumMatrix(matrix)
    assert nm.sum_region(2, 1, 4, 3) == 8
    assert nm.sum_region(1, 1, 2, 2) == 11
    assert nm.sum_region(0, 0, 0, 0) == 3
    assert nm.sum_region(4, 4, 4, 4) == 5
    assert nm.sum_region(0, 0, 4, 4) == 58
    single = NumMatrix([[7]])
    assert single.sum_region(0, 0, 0, 0) == 7
    print("range_sum_query_2d: all tests passed")
