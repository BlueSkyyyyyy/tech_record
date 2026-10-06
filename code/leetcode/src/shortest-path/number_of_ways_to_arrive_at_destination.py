"""1976. 到达目的地的方案数（Number of Ways to Arrive at Destination）

题目：城市编号 0~n-1，roads[i] = [u, v, time] 表示 u 与 v 之间有一条双向道路，
通行耗时 time。求从 0 号城市到 n-1 号城市、在「总耗时最短」的前提下，有多少条
不同的路径。答案对 1e9+7 取模。

思路（Dijkstra + 路径计数）：
    先求最短路，再数「有多少条路能凑出这个最短路」。用两个数组：
        dist[v]：0 到 v 的最短耗时；
        ways[v]：0 到 v 的「最短路径条数」。
    初始化 dist[0]=0、ways[0]=1。Dijkstra 扩展边 u->v（耗时 w）时：
        - 若 dist[u]+w < dist[v]：发现更短的路，dist[v] 更新，ways[v] = ways[u]；
        - 若 dist[u]+w == dist[v]：又找到一条等长的路，ways[v] += ways[u]。
    因为所有要被累加的前驱 u 都在 v 之前出堆（边权非负），到 v 出堆时 ways[v] 已经
    数完所有最短前驱。

    为什么用 Dijkstra 而不是普通 DP：图是一般图（有环、边权不同），最短路的「前驱」
    顺序不能靠下标，只能靠 Dijkstra 按距离从小到大确定。

复杂度：时间 O(E log V)，空间 O(V + E)。
"""
import heapq
from collections import defaultdict

_MOD = 10 ** 9 + 7


def count_paths(n, roads):
    graph = defaultdict(list)
    for u, v, w in roads:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dist = [INF] * n
    ways = [0] * n
    dist[0] = 0
    ways[0] = 1
    heap = [(0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                ways[v] = ways[u]
                heapq.heappush(heap, (nd, v))
            elif nd == dist[v]:
                ways[v] = (ways[v] + ways[u]) % _MOD
    return ways[n - 1] % _MOD


if __name__ == "__main__":
    # 两条等长的最短路：0->1->3 与 0->2->3
    assert count_paths(4, [[0, 1, 1], [0, 2, 1], [1, 3, 1], [2, 3, 1]]) == 2
    # 直接边 5 太慢，唯一最短路是 0->1->2
    assert count_paths(3, [[0, 1, 1], [1, 2, 1], [0, 2, 5]]) == 1
    assert count_paths(2, [[0, 1, 1]]) == 1
    # 三条等长的最短路：直达、经 1、经 2，耗时都是 2
    assert (
        count_paths(
            4,
            [
                [0, 3, 2], [0, 1, 1], [1, 3, 1], [0, 2, 1], [2, 3, 1],
            ],
        )
        == 3
    )
    print("count_paths: all tests passed")
