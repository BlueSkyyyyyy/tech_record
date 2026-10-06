"""542. 01 矩阵（01 Matrix）

题目：给定一个由 0 和 1 组成的矩阵 mat，请输出一个大小相同的矩阵，
其中每个格子是 mat 中对应位置元素到最近的 0 的距离。相邻元素之间的距离为 1。

思路（多源 BFS，从所有 0 同时向外扩散）：
    直觉上可以对每个 1 单独 BFS 找最近的 0，但那样每个 1 都要重新搜一遍，代价高。
    反过来想：所有 0 都是距离为 0 的源，同时向外扩散一层，遇到的 1 距离就是 1，
    再往外一层是 2……于是只要做一次「多源 BFS」：

    1. 开一个 dist 矩阵，初始全为 -1（表示未确定）；把所有 0 的位置设成 0 并入队；
    2. 从队列逐个取出格子，检查四个邻居：若邻居还没确定距离，
       它的距离就等于当前格距离加一，入队；
    3. 队列跑空后，每个格子的最短距离就都填好了。

    为什么一次多源 BFS 就够：BFS 保证按距离从小到大访问；多个源同时入队相当于
    在图上加了一个虚拟源点连向所有 0，问题变成「虚拟源点到每个格子的最短距离」，
    一次 BFS 即可。为什么条件是 dist == -1：只有第一次访问到某个格子时，
    才是离它最近的那个 0 扩散过来的，之后再访问不会更短。

复杂度：时间 O(m·n)（每个格子至多入队一次），空间 O(m·n)（dist + 队列）。
"""
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def update_matrix(mat):
    rows, cols = len(mat), len(mat[0])
    dist = [[-1] * cols for _ in range(rows)]
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dist[r][c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist


if __name__ == "__main__":
    assert update_matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    assert update_matrix([[0, 0, 0], [0, 1, 0], [1, 1, 1]]) == [
        [0, 0, 0],
        [0, 1, 0],
        [1, 2, 1],
    ]
    # 单个格子且为 0
    assert update_matrix([[0]]) == [[0]]
    # 全 1，没有 0 作源（题目保证至少有一个 0），距离保持未确定
    assert update_matrix([[1, 1], [1, 1]]) == [[-1, -1], [-1, -1]]
    print("update_matrix: all tests passed")
