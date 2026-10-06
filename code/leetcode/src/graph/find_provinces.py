"""547. 省份数量（Number of Provinces）

题目：有 n 个城市，其中一些彼此相连、另一些没有。城市 i 与城市 j 直接相连记作
isConnected[i][j] == 1。如果城市 a 与 b 直接相连、b 与 c 直接相连，
那么 a 与 c 也间接相连。省份是一组直接或间接相连的城市，组内不含其它未相连的城市。
返回省份的数量。

思路（并查集：把相连的城市合并成一个集合）：
    「有多少个互不相连的组」就是「有多少个连通块」，用并查集最合适。
    并查集维护若干不相交的集合，支持两种操作：
      - find(x)：找到 x 所在集合的代表（根）；
      - union(a, b)：把 a、b 所在的两个集合合并成一个。
    初始时每个城市各自是一个省份，省份数 count = n。
    遍历矩阵的上三角（i < j），凡是 isConnected[i][j] == 1 就让两个城市 union：
      - 若两者本就同根，说明已在同一省份，什么也不做；
      - 否则合并两个集合，省份数减一（两个省并成了一个）。
    遍历结束后 count 就是答案。

    为什么从 n 开始做减法：每成功合并一次，独立的集合就少一个，天然得到省份数，
    不必再对每个城市 find 一遍去去重。

    路径压缩（parent[x] = parent[parent[x]]）把树压扁，后续查找接近 O(1)，
    否则反复合并可能退化成链。

复杂度：时间 O(n^2 · α(n))，α 是反阿克曼函数（近似常数），主要开销在遍历 n x n 矩阵；
    空间 O(n)（parent 数组）。
"""


def find_circle_num(is_connected):
    n = len(is_connected)
    parent = list(range(n))
    count = n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    count -= 1
    return count


if __name__ == "__main__":
    assert find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]) == 2
    assert find_circle_num([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 3
    assert find_circle_num([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) == 1
    assert find_circle_num([[1]]) == 1
    print("find_provinces: all tests passed")
