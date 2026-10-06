"""1368. 使网格图至少有一条有效路径的最小代价（Minimum Cost to Make at Least One
Valid Path in a Grid）

题目：给一个 m×n 的网格，每格的数字表示一个「路标方向」：
1=向右，2=向左，3=向下，4=向上。你可以修改任意格的路标，每修改一格代价为 1。
从左上角 (0,0) 出发，只能沿所站格子的路标方向移动。求让 (m-1,n-1) 可达的最小代价。

思路（边权只有 0 和 1 → 0-1 BFS）：
    从某格 (r,c) 走向它的四个邻居，如果邻居恰好是 (r,c) 路标指向的方向，代价 0；
    否则要改这格的路标，代价 1。于是得到一张边权为 0 或 1 的图，求最短路。

    普通的队列 BFS 只能处理边权全 1；Dijkstra 能处理 0/1 但要带堆。既然边权非 0
    即 1，有个更省的双端队列 BFS（0-1 BFS）：
        - 走 0 权边，把新节点压到队首（同层，保持距离不增）；
        - 走 1 权边，把新节点压到队尾。
    这样队列里节点距离最多只差 1、且按距离非递减，出队顺序就等价于 Dijkstra。
    仍用 dist[nr][nc] > nd 判断能否松弛（一个节点可能被更优地再入队一次）。

复杂度：时间 O(m·n)（每个节点至多入队两次），空间 O(m·n)。
"""
from collections import deque

# 下标 0/1/2/3 分别对应路标 1/2/3/4：右、左、下、上
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def min_cost(grid):
    m, n = len(grid), len(grid[0])
    INF = float("inf")
    dist = [[INF] * n for _ in range(m)]
    dist[0][0] = 0
    dq = deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        for i, (dr, dc) in enumerate(_DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                cost = 0 if grid[r][c] == i + 1 else 1
                nd = dist[r][c] + cost
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
    return dist[m - 1][n - 1]


if __name__ == "__main__":
    # 蛇形排布：每换一行都要改一次方向，共 3 次
    assert min_cost([[1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1, 1], [2, 2, 2, 2]]) == 3
    assert min_cost([[1, 1, 3], [3, 2, 2], [1, 1, 4]]) == 0
    assert min_cost([[1, 2], [4, 3]]) == 1
    # 全指向左，每一步都与目标方向不符，共 2 次修改
    assert min_cost([[2, 2], [2, 2]]) == 2
    print("min_cost: all tests passed")
