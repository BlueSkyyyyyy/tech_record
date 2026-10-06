"""498. 对角线遍历（Diagonal Traverse）

题目：给定 m x n 矩阵 mat，按对角线顺序遍历所有元素，要求相邻两条对角线的
行进方向交替（第 1 条向右上，第 2 条向左下，第 3 条向右上……），返回遍历序列。

思路（按「行下标 + 列下标」枚举对角线）：
    同一条对角线上的元素满足「行下标 + 列下标 = d」，d 从 0 到 m+n-2，共 m+n-1 条。
    对每条对角线，先算出它进入矩阵的起点，再顺着方向走：
      - d 为偶数：从左下往右上；
      - d 为奇数：从右上往左下。
    起点要保证不越界：向上走时让行下标取 min(d, m-1)、列下标为 d 减它；向下走时
    让列下标取 min(d, n-1)、行下标为 d 减它。

    为什么用 d 而不是逐格判断方向：d 的奇偶天然对应方向，且每条对角线独立，
    不必维护一个全局的「上一步往哪走」，边界条件更少、更不容易错。
"""


def find_diagonal_order(mat):
    m, n = len(mat), len(mat[0])
    res = []
    for d in range(m + n - 1):
        if d % 2 == 0:
            r = min(d, m - 1)
            c = d - r
            while r >= 0 and c < n:
                res.append(mat[r][c])
                r -= 1
                c += 1
        else:
            c = min(d, n - 1)
            r = d - c
            while c >= 0 and r < m:
                res.append(mat[r][c])
                r += 1
                c -= 1
    return res


if __name__ == "__main__":
    assert find_diagonal_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [
        1, 2, 4, 7, 5, 3, 6, 8, 9]
    assert find_diagonal_order([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert find_diagonal_order([[1, 2, 3]]) == [1, 2, 3]
    assert find_diagonal_order([[1], [2], [3]]) == [1, 2, 3]
    assert find_diagonal_order([[1, 2, 3, 4], [5, 6, 7, 8]]) == [
        1, 2, 5, 6, 3, 4, 7, 8]
    print("find_diagonal_order: all tests passed")
