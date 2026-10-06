"""1334. 阈值距离内邻居最少的城市（Find the City With the Smallest Number of
Neighbors at a Threshold Distance）

题目：有 n 个城市，edges[i] = [u, v, w] 表示 u、v 间的双向道路长 w。给定距离阈值
distanceThreshold，若城市 i 到城市 j 的最短距离不超过该阈值，就称 j 是 i 的邻居。
求「邻居数量最少」的城市；若有多个，返回编号最大的那个，且要求存在至少一个邻居。

思路（任意两点间最短路 → Floyd-Warshall）：
    本题要对**每个城市**都求出到其他所有城市的最短路（多源 / 全源最短路），
    Floyd-Warshall 正合适。它用「允许经过哪些中转点」逐层放松：
        for k: for i: for j:
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    第 k 轮结束时，dist[i][j] 表示「只允许经过编号 ≤ k 的点」的最短路；k 扫完就是
    全局最短。

    注意三重循环的顺序：中转点 k 必须在最外层。把 k 放内层会过早固化某些中转，
    得到错误结果。

    最后统计每个城市在阈值内的邻居数，取最少者；用 `<=` 比较即可在并列时自动选择
    编号更大的城市（因为按编号递增扫描）。若某城市没有任何满足条件的邻居，不计入。

复杂度：时间 O(n³)，空间 O(n²)。
"""


def find_the_city(n, edges, distance_threshold):
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
        if w < dist[v][u]:
            dist[v][u] = w

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    best_city = -1
    best_count = n + 1
    for i in range(n):
        cnt = sum(
            1 for j in range(n) if i != j and dist[i][j] <= distance_threshold
        )
        if cnt <= best_count:
            best_count = cnt
            best_city = i
    return best_city


if __name__ == "__main__":
    assert (
        find_the_city(4, [[0, 1, 3], [1, 2, 1], [1, 3, 4], [2, 3, 1]], 4) == 3
    )
    assert (
        find_the_city(
            5,
            [[0, 1, 2], [0, 4, 8], [1, 2, 3], [1, 4, 2], [2, 3, 1], [3, 4, 1]],
            2,
        )
        == 0
    )
    print("find_the_city: all tests passed")
