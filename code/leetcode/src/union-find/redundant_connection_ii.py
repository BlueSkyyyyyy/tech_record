"""685. 冗余连接 II（Redundant Connection II）

题目：给一个有向图，它由一棵「以某节点为根的有向树」多加一条边得到。找出那条可以
删掉、使图重新变回有向树的边；若有多个答案，返回输入中最后出现的那条。节点编号 1..n。

思路（有向图分两种破坏方式，先找「双父」再判环）：
    一棵以某点为根的有向树有两个条件：除根外每个节点**入度为 1**，且无环。
    多加一条边只会破坏其中一个（或两个都破坏）：
    - 某个节点入度为 2（有两条边指向它），这是「双父」；
    - 出现一个有向环。

    分情况：
    1. 没有双父：那问题一定是环，删掉使环闭合的那条边即可，退化成本篇 684；
    2. 有双父（记第二条指向 v 的边为 conflict）：答案必是这 v 的两条入边之一。
       先假设删掉较晚的 conflict 边，看看图里还有没有环：
       - 没有环 → 删 conflict 即可；
       - 仍有环 → 说明 conflict 不是环上的边，必须删掉较早的那条入边。

    实现上先照常并查集扫一遍，遇到双父时**不**把 conflict 边并入（先把它当作待删），
    同时记录是否有环（union 失败即环）；最后按上面的规则给出答案。

复杂度：时间 O(n·α(n))，空间 O(n)。
"""
from dsu import DSU


def find_redundant_directed_connection(edges):
    n = len(edges)
    dsu = DSU(n + 1)
    parent = [0] * (n + 1)
    conflict = -1
    cycle = -1

    for i, (u, v) in enumerate(edges):
        if parent[v] != 0:
            conflict = i
        else:
            parent[v] = u
            if not dsu.union(u, v):
                cycle = i

    if conflict == -1:
        return list(edges[cycle])
    if cycle == -1:
        return list(edges[conflict])
    return [parent[edges[conflict][1]], edges[conflict][1]]


if __name__ == "__main__":
    assert find_redundant_directed_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_directed_connection([[1, 2], [2, 3], [3, 4], [4, 1], [1, 5]]) == [4, 1]
    assert find_redundant_directed_connection([[2, 1], [3, 1], [4, 2], [1, 4]]) == [2, 1]
    print("redundant_connection_ii: all tests passed")
