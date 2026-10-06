"""1928. 规定时间内到达终点的最小花费（Minimum Cost to Reach Destination in Time）

题目：n 个城市，edges[i] = [u, v, time] 表示 u、v 间双向通行耗时 time。
passingFees[i] 是「经过城市 i」要交的过路费（起点和终点也要交）。你从城市 0 出发，
必须在总耗时不超过 maxTime 的前提下到达城市 n-1，求最小总花费；无法在时限内到达
返回 -1。

思路（把「时间」也当作状态 → DP / 分层图最短路）：
    如果只记「到城市 v 的最小花费」，这个状态不够用：花费小的路径可能太慢，后面
    反而到不了终点。所以要把时间一起放进状态：
        dp[t][v] = 在恰好（不超过）耗时 t 到达 v 的最小花费。
    转移：从 (t, u) 出发走边 (u, v, w)，到达 (t+w, v)，花费加上 v 的门票：
        dp[t+w][v] = min(dp[t+w][v], dp[t][u] + passingFees[v])
    初始 dp[0][0] = passingFees[0]（起点也要交费）。按 t 从小到大遍历即可，因为
    转移总把时间推大，天然满足拓扑序。

    这样做等价于在一张「(城市, 时间)」的分层图上跑最短路（时间维度只在 0~maxTime
    之间、且边权非负，所以也可以看成按时间递推的 DP）。

复杂度：时间 O(maxTime · E)，空间 O(maxTime · n)。
"""


def min_cost(max_time, edges, passing_fees):
    n = len(passing_fees)
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dp = [[INF] * n for _ in range(max_time + 1)]
    dp[0][0] = passing_fees[0]
    for t in range(max_time + 1):
        for u in range(n):
            if dp[t][u] == INF:
                continue
            for v, w in graph[u]:
                nt = t + w
                if nt <= max_time and dp[t][u] + passing_fees[v] < dp[nt][v]:
                    dp[nt][v] = dp[t][u] + passing_fees[v]

    ans = min(dp[t][n - 1] for t in range(max_time + 1))
    return -1 if ans == INF else ans


if __name__ == "__main__":
    # 直达花 7、耗时 30；中转花 8、耗时 20，取更便宜的 7
    assert min_cost(30, [[0, 1, 10], [1, 2, 10], [0, 2, 30]], [5, 1, 2]) == 7
    assert min_cost(5, [[0, 1, 5]], [1, 3]) == 4
    # 时限不够，到不了终点
    assert min_cost(4, [[0, 1, 5]], [1, 3]) == -1
    print("min_cost: all tests passed")
