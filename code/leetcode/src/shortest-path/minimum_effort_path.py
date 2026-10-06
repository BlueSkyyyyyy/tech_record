"""1631. 最小体力消耗路径（Path With Minimum Effort）

题目：给一个 m×n 的高度矩阵 heights。从左上角走到右下角，每一步走到相邻（上下
左右）格子，一条路径的「体力消耗」定义为路径上所有相邻格高度差绝对值的最大值。
求所有路径中最小的体力消耗。

思路（瓶颈路 → 改造版 Dijkstra）：
    普通最短路把路径代价定义为「边权之和」；本题定义的却是「路径上最大的一条边」，
    这类问题叫瓶颈路（minimax path）。

    改造 Dijkstra 的松弛公式：
        dist[v] = min( dist[v], max(dist[u], |h[v] - h[u]|) )
    意思是：到 v 的这条瓶颈，等于「到 u 的瓶颈」和「u-v 这条边的落差」中较大的
    那一个。仍然用优先队列每次取瓶颈最小的节点扩展，同理可证它已最优。

    为什么可行：max 关于非负边权保持单调——沿路径加边不会让瓶颈变小；于是取当前
    瓶颈最小的节点扩展，与 Dijkstra 的贪心论证完全一致。

复杂度：时间 O(m·n·log(m·n))，空间 O(m·n)。
"""
import heapq

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def minimum_effort_path(heights):
    m, n = len(heights), len(heights[0])
    INF = float("inf")
    dist = [[INF] * n for _ in range(m)]
    dist[0][0] = 0
    heap = [(0, 0, 0)]
    while heap:
        effort, r, c = heapq.heappop(heap)
        if r == m - 1 and c == n - 1:
            return effort
        if effort > dist[r][c]:
            continue
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                ne = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if ne < dist[nr][nc]:
                    dist[nr][nc] = ne
                    heapq.heappush(heap, (ne, nr, nc))
    return dist[m - 1][n - 1]


if __name__ == "__main__":
    assert minimum_effort_path([[1, 2, 2], [3, 8, 2], [5, 3, 5]]) == 2
    assert minimum_effort_path([[1, 2, 3], [3, 8, 4], [5, 3, 5]]) == 1
    assert (
        minimum_effort_path(
            [
                [1, 2, 1, 1, 1],
                [1, 2, 1, 2, 1],
                [1, 2, 1, 2, 1],
                [1, 2, 1, 2, 1],
                [1, 1, 1, 2, 1],
            ]
        )
        == 0
    )
    print("minimum_effort_path: all tests passed")
