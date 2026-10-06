"""133. 克隆图（Clone Graph）

题目：给你无向连通图中一个节点的引用 node，请你返回该图的深拷贝（克隆）。
每个节点的值都等于它的编号，节点用 neighbors 列表保存所有邻居的引用。

思路（DFS + 哈希表建立「原节点 -> 克隆节点」映射）：
    深拷贝的关键是：同一个原节点只能克隆一次，否则环上的节点会被反复创建、
    克隆图不再与原图同构。于是开一个字典 clones，键是原节点、值是它的克隆。
    从 node 出发做 DFS：
      - 若当前节点已在 clones 里，直接返回它对应的克隆，避免重复创建；
      - 否则先新建一个只有值、没有邻居的克隆并登记，再逐个克隆它的邻居，
        把克隆出的邻居接到克隆节点的 neighbors 上。
    先登记、后递归邻居，是为了处理「自环」和「环」：
    递归回到自己时能从字典里取到已经建好的克隆，而不会无限递归。

复杂度：时间 O(V + E)（每个节点、每条边各访问一次），空间 O(V)（哈希表 + 递归栈）。
"""


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    if node is None:
        return None

    clones = {}

    def dfs(cur):
        if cur in clones:
            return clones[cur]
        copy = Node(cur.val)
        clones[cur] = copy
        for nb in cur.neighbors:
            copy.neighbors.append(dfs(nb))
        return copy

    return dfs(node)


def _build(adj):
    """由邻接表（编号从 1 开始）构造图，返回编号 1 的节点。"""
    if not adj:
        return None
    nodes = {i: Node(i) for i in range(1, len(adj) + 1)}
    for i, nbrs in enumerate(adj, start=1):
        nodes[i].neighbors = [nodes[j] for j in nbrs]
    return nodes[1]


def _to_adj(node):
    """把图序列化成邻接表，便于比较（按编号排序）。"""
    if node is None:
        return []
    seen = {}
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.val in seen:
            continue
        seen[cur.val] = cur
        for nb in cur.neighbors:
            if nb.val not in seen:
                stack.append(nb)
    return [sorted(nb.val for nb in seen[v].neighbors) for v in sorted(seen)]


if __name__ == "__main__":
    adj = [[2, 4], [1, 3], [2, 4], [1, 3]]
    original = _build(adj)
    cloned = clone_graph(original)

    assert cloned is not original
    assert _to_adj(cloned) == adj
    # 深拷贝：改克隆图不影响原图
    cloned.neighbors[0].val = 99
    assert original.neighbors[0].val == 2

    assert clone_graph(None) is None
    single = Node(1)
    one = clone_graph(single)
    assert one.val == 1 and one is not single and one.neighbors == []

    print("clone_graph: all tests passed")
