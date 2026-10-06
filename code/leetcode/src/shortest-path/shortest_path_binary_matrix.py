"""1091. 二进制矩阵中的最短路径（Shortest Path in Binary Matrix）

题目：给一个 n×n 的 0/1 矩阵 grid，0 表示空地、1 表示障碍。从左上角 (0,0) 出发，
每一步可以走到「上下左右 + 四个斜角」共 8 个相邻的 0 格，求到达右下角 (n-1,n-1)
的最短路径长度（经过的格子数）。无路可达返回 -1。

思路（边权全为 1 的图 → 普通 BFS）：
    每条边的代价都是 1（走一步），这是最朴素的最短路场景：BFS 逐层扩展，
    第一次到达某格时走过的步数就是它到起点的最短距离，无需 Dijkstra。

    具体做法：
    1. 起点或终点是障碍，直接返回 -1；
    2. dist[r][c] 记录「经过的格子数」，起点为 1（起点自身算一格）；
    3. 队列 BFS，八个方向扩展；只走 0 格、且未访问过（dist 为 0）；
    4. 一旦弹出终点即返回其 dist。

    为什么 BFS 就是最短路：队列按距离分层，距离为 d 的格子一定在距离为 d+1 的
    格子之前出队；所以某个格子第一次被访问到，用的就是最少步数。

复杂度：时间 O(n²)（每格至多入队一次），空间 O(n²)。
"""
from collections import deque

_DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1
    dist = [[0] * n for _ in range(n)]
    dist[0][0] = 1
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        if r == n - 1 and c == n - 1:
            return dist[r][c]
        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and dist[nr][nc] == 0:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return -1


if __name__ == "__main__":
    assert shortest_path_binary_matrix([[0, 1], [1, 0]]) == 2
    assert shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
    assert shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]) == -1
    assert shortest_path_binary_matrix([[0]]) == 1
    assert shortest_path_binary_matrix([[1]]) == -1
    # 斜着一步就能到达终点
    assert shortest_path_binary_matrix([[0, 0], [0, 0]]) == 2
    print("shortest_path_binary_matrix: all tests passed")
