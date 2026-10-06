"""766. 托普利茨矩阵（Toeplitz Matrix）

题目：如果一个矩阵的每一条从左上到右下的对角线上的元素都相同，就称它是托普
利茨矩阵。给定矩阵，判断它是否满足这一性质。

思路（每个元素和它的左上邻居比）：
    对每个非首行、非首列的元素 matrix[i][j]，它都应该和左上角的
    matrix[i-1][j-1] 相等。一旦有一处不等，立即返回 False；全部通过则为 True。

    为什么只比左上邻居就够：同一条对角线上的元素满足「i - j 相同」，相邻两个
    位置正好是 (i-1, j-1) 和 (i, j)。只要每一对相邻都相等，由相等的传递性，
    整条对角线自然都相等，不必两两比较。

复杂度：时间 O(m*n)，空间 O(1)。
"""


def is_toeplitz_matrix(matrix):
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True


if __name__ == "__main__":
    assert is_toeplitz_matrix([[1, 2, 3, 4], [5, 1, 2, 3], [9, 5, 1, 2]])
    assert not is_toeplitz_matrix([[1, 2], [2, 2]])
    assert is_toeplitz_matrix([[1]])
    assert is_toeplitz_matrix([[1, 2], [3, 1]])
    assert not is_toeplitz_matrix([[1, 2, 3], [4, 5, 1]])
    print("is_toeplitz_matrix: all tests passed")
