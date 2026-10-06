"""787. K 站中转内最便宜的航班（Cheapest Flights Within K Stops）

题目：有 n 个城市，flights[i] = [from, to, price] 表示一条有向航线。求从 src 到 dst
在「最多经过 k 个中转站」限制下的最便宜价格；不可达返回 -1。

思路（限定边数 → Bellman-Ford 风格按轮松弛）：
    「最多 k 个中转站」等价于「路径至多 k+1 条边」。这类「限定边数的单源最短路」
    正是 Bellman-Ford 的用武之地：做 k+1 轮松弛，第 i 轮结束后的 dist 恰好表示
    「用不超过 i 条边能到达的最低价格」。

    关键点：每一轮必须基于**上一轮的 dist 快照**去松弛，而不能在本轮已经更新的
    结果上继续松弛。否则同一条路径会在一轮内被反复延长，等于使用了任意多条边，
    就绕过了 k 的限制。所以每轮先 `prev = dist[:]`，松弛时读 prev、写 dist。

    为什么能覆盖负权/一般图：Bellman-Ford 不要求边权非负，也不要求图无环，它只是
    「最多 i 条边的最优值」的 DP 递推。

复杂度：时间 O(k·E)，空间 O(V)。
"""


def find_cheapest_price(n, flights, src, dst, k):
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        prev = dist[:]
        for u, v, w in flights:
            if prev[u] != INF and prev[u] + w < dist[v]:
                dist[v] = prev[u] + w
    return -1 if dist[dst] == INF else dist[dst]


if __name__ == "__main__":
    flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
    # 最多 1 个中转：0->1->3 = 700；0->1->2->3 用了 2 个中转，不允许
    assert find_cheapest_price(4, flights, 0, 3, 1) == 700
    flights2 = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
    # 1 个中转可以走 0->1->2，200 更便宜
    assert find_cheapest_price(3, flights2, 0, 2, 1) == 200
    # 不允许中转时只能坐直飞
    assert find_cheapest_price(3, flights2, 0, 2, 0) == 500
    # 没有任何从 0 到 1 的航线
    assert find_cheapest_price(2, [[1, 0, 100]], 0, 1, 1) == -1
    print("find_cheapest_price: all tests passed")
