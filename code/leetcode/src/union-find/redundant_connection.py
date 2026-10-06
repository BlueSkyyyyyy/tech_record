"""684. 冗余连接（Redundant Connection）

题目：一棵 n 个节点的树本来没有环，现在多了一条边，变成一个带恰一条环的图。
给定 n 条边（无向，节点编号 1..n），找出那条「可以删掉、使图重新变回树」的边。
若有多条，返回输入中最后出现的那条。

思路（并查集判环）：
    逐条处理边 (a, b)。若 a、b 此刻还不在同一个连通块里，说明这条边安全，union 之；
    若 a、b 已经连通，那么再加这条边就会成环——它就是答案。

    为什么是「最后出现」：按输入顺序处理，第一次遇到「两端已连通」的边时，
    它前面所有边都还没成环且已构成一个连通块，所以这条就是使环闭合的边。
    题目保证恰有一条环，因此第一条冲突边必是答案。

复杂度：时间 O(n·α(n))，空间 O(n)。
"""
from dsu import DSU


def find_redundant_connection(edges):
    dsu = DSU(len(edges) + 1)
    for a, b in edges:
        if not dsu.union(a, b):
            return [a, b]
    return []


if __name__ == "__main__":
    assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_connection([[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]) == [1, 4]
    assert find_redundant_connection([[1, 2], [2, 3], [1, 3]]) == [1, 3]
    print("redundant_connection: all tests passed")
