"""743. 网络延迟时间（Network Delay Time）

题目：有 n 个网络节点，编号 1~n。给定有向边 times[i] = [u, v, w]，表示信号从 u 传到 v
耗时 w。现在从节点 k 发出一个信号，求所有节点都收到信号所需的最短时间；若有节点
收不到，返回 -1。

思路（非负边权单源最短路 → Dijkstra）：
    所有边权非负，从源点 k 到每个节点的最短距离可以用 Dijkstra 求。答案就是
    「所有节点最短距离的最大值」——因为信号沿各自最短路径传播，最后一个节点收到
    的时刻就是最长的那条最短路径。若有节点不可达（距离仍为无穷），返回 -1。

    Dijkstra 的核心是贪心：每次从未确定点里取出「当前距离最小」的节点 u，此时
    dist[u] 已经是最优的（不可能再经由别的未确定点变得更短，因为那些点的距离都
    不小于 dist[u]，而边权非负）。以 u 为中转去松弛它的邻居 v：若 dist[u]+w 更小，
    就更新 dist[v] 并入堆。

    用优先队列（小顶堆）取出最小距离；堆里可能有过期条目，弹出时用 d > dist[u]
    判断跳过即可。

复杂度：时间 O(E log V)（每条边可能入堆一次），空间 O(V + E)。
"""
import heapq
from collections import defaultdict


def network_delay_time(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    INF = float("inf")
    dist = [INF] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    ans = max(dist[1:])
    return -1 if ans == INF else ans


if __name__ == "__main__":
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    # 源点无法到达节点 1
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1
    # 需要绕路：直接边很慢，中转更快
    assert network_delay_time([[1, 2, 1], [2, 3, 1], [1, 3, 5]], 3, 1) == 2
    print("network_delay_time: all tests passed")
