"""847. 访问所有节点的最短路径（Shortest Path Visiting All Nodes）

题目：给定一个连通的无向图（邻接表 graph），求访问所有节点所需的最短路径长度。
可以任意起点、任意终点，节点和边可以重复经过。

思路（状态 = (当前所在节点, 已经访问过的节点集合)）：
    普通的 BFS 只记"我在哪个点"，但本题要保证"所有点都访问过"，所以状态必须带上
    "访问过哪些点"。n ≤ 12，访问集合用 n 位掩码表示，状态总数 ≤ n * 2^n。

    把起点设为任意一个节点（把它对应的掩码位置 1），所有起点一起入队做多源 BFS。
    每一步沿着一条边走：走到 v 时把 v 也并入访问集合。第一次到达"掩码 = 全集"的状态，
    步数就是答案——因为 BFS 按层扩展，最先到的一定最短。

    这就是"状态压缩最短路"：把"集合信息"压进状态的一位，把最短路问题变成图上 BFS。

复杂度：时间 O(2^n * (n + m))，空间 O(2^n * n)。
"""

from collections import deque


def shortest_path_length(graph):
    n = len(graph)
    full = (1 << n) - 1

    dist = [[-1] * n for _ in range(1 << n)]
    queue = deque()
    for i in range(n):
        mask = 1 << i
        dist[mask][i] = 0
        queue.append((mask, i))

    while queue:
        mask, u = queue.popleft()
        d = dist[mask][u]
        if mask == full:
            return d
        for v in graph[u]:
            nxt = mask | (1 << v)
            if dist[nxt][v] == -1:
                dist[nxt][v] = d + 1
                queue.append((nxt, v))
    return -1


if __name__ == "__main__":
    assert shortest_path_length([[1, 2, 3], [0], [0], [0]]) == 4
    assert shortest_path_length([[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]) == 4
    assert shortest_path_length([[]]) == 0
    assert shortest_path_length([[1], [0]]) == 1
    print("shortest_path_length: all tests passed")
