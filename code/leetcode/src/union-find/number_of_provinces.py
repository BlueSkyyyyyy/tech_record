"""547. 省份数量（Number of Provinces）

题目：给一个 n×n 的矩阵 is_connected，is_connected[i][j] == 1 表示城市 i 与 j 直接相连
（连通具有传递性）。互相直接或间接连通的城市构成一个「省份」，求省份总数。

思路（并查集求连通分量个数）：
    把每个城市看成一个点，矩阵里的每条「1」看成一条无向边。省份数就是这张图的
    连通分量个数。用并查集：遍历矩阵上三角，只要 i、j 相连就 union 一下；
    最后数一数有多少个不同的根，就是连通分量个数。

    为什么只扫上三角：矩阵是对称的，i-j 与 j-i 是同一条边，扫一遍就够。

复杂度：时间近似 O(n²·α(n))（主要是读矩阵），空间 O(n)。
"""
from dsu import DSU


def find_circle_num(is_connected):
    n = len(is_connected)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                dsu.union(i, j)
    return len({dsu.find(i) for i in range(n)})


if __name__ == "__main__":
    assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
    assert find_circle_num([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3
    assert find_circle_num([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 1
    assert find_circle_num([[1]]) == 1
    print("number_of_provinces: all tests passed")
