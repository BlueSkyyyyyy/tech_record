"""684. 冗余连接（Redundant Connection）

题目：有一棵 n 个节点（编号 1 ~ n）的树，原本无环。现在额外添加了一条边，
使得图中出现了一个环。给你边的数组 edges（edges[i] = [u, v] 表示节点 u 与 v 之间
有一条边），请找出那条可以删去的边，使得剩下的图是一棵有 n 个节点的树。
如果有多个答案，返回输入中最后出现的那条。

思路（并查集：按输入顺序加边，第一条让两端已连通的边就是答案）：
    树的特点是「任意两点之间恰好一条路径、没有环」。逐条把边加进并查集：
      - 如果一条边的两个端点当前不在同一集合，说明它把两个连通块连了起来，
        这条边是必要的，执行 union；
      - 如果两个端点已经在同一集合，说明它们之间原本就有路径，
        再加上这条边就会形成环——它正是那条多余的边。

    为什么返回「最后出现」的那条能自动满足：我们是从前往后逐条检查，
    一旦发现加入会成环就立刻返回。题目保证只多了一条边、环唯一，
    所以第一条（也是唯一一条）造成环的边就是答案；若有多解，
    按输入顺序检查得到的就是最后出现的那条。

复杂度：时间 O(n · α(n))（近似 O(n)，每条边一次 find/union），空间 O(n)。
"""


def find_redundant_connection(edges):
    n = len(edges)
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return [u, v]
        parent[ru] = rv
    return []


if __name__ == "__main__":
    assert find_redundant_connection([[1, 2], [1, 3], [2, 3]]) == [2, 3]
    assert find_redundant_connection(
        [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]
    ) == [1, 4]
    assert find_redundant_connection([[1, 2], [2, 3], [1, 3]]) == [1, 3]
    print("redundant_connection: all tests passed")
