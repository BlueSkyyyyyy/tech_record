"""785. 判断二分图（Is Graph Bipartite?）

题目：给定无向图的邻接表 graph，判断能否把节点分成两个集合，使每条边的两端都落在
不同集合（即图可以二染色）。

思路（「敌人的敌人是朋友」并查集版）：
    二分图的定义等价于：对任意节点 u，它的所有邻居必须站在 u 的对立面，因此
    **u 的所有邻居彼此必须处在同一阵营**。于是：
    - 遍历每个节点 u，把它的所有邻居都 union 到一起；
    - 若过程中发现 u 和某个邻居已经在同一集合，说明出现了两条「同色」相邻的边，
      不可能二染色，返回 False。

    为什么这样对：把「同一集合」理解为同色。邻居们都被规定成异于 u 的颜色，
    即彼此同色；一旦某个邻居和 u 撞进同色集合就矛盾。

    这与 DFS 染色是两种等价视角：DFS 显式地给每个点染色并检查冲突，
    并查集则隐式地维护「谁和谁同色」。

复杂度：时间 O(E·α(V))（E 为边数，V 为点数），空间 O(V)。
"""
from dsu import DSU


def is_bipartite(graph):
    n = len(graph)
    dsu = DSU(n)
    for u in range(n):
        for v in graph[u]:
            if dsu.find(u) == dsu.find(v):
                return False
            dsu.union(graph[u][0], v)
    return True


if __name__ == "__main__":
    assert is_bipartite([[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]) is False
    assert is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]]) is True
    assert is_bipartite([[]]) is True
    assert is_bipartite([[1], [0]]) is True
    assert is_bipartite([[1, 2], [0], [0]]) is True
    print("is_graph_bipartite: all tests passed")
