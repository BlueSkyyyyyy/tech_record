"""994. 腐烂的橘子（Rotting Oranges）

题目：在给定的 m x n 网格 grid 中，每个单元格可能有三种值：
0 表示空格、1 表示新鲜橘子、2 表示腐烂橘子。每分钟，腐烂橘子会把它上下左右
相邻的新鲜橘子变腐烂。返回直到单元格中没有新鲜橘子为止所必须经过的最小分钟数；
如果不可能全部腐烂，返回 -1。

思路（多源 BFS，把「同时扩散」的所有源一起入队）：
    这是「最短时间」问题，应当用 BFS 而不是 DFS——BFS 天然按层扩散，每一层恰好对应一分钟。
    关键点是「一开始就有多个腐烂橘子」，它们在同一分钟同时向外扩散，
    所以要把所有腐烂橘子**一起**作为第 0 层的源放进队列，这就是「多源 BFS」。

    流程：
    1. 扫描网格，把所有腐烂橘子入队，同时数出新鲜橘子的数量 fresh；
    2. 每一轮循环处理「当前队列里的全部节点」，它们代表同一分钟被感染的橘子；
       每处理一个就检查四个邻居，把新鲜邻居变腐烂、fresh 减一、入队；
    3. 一轮结束后分钟数加一；
    4. 循环结束时若 fresh == 0，返回分钟数；否则说明有橘子被孤立，返回 -1。

    为什么要按「当前队列长度」分层：只有把同一分钟的扩散集中处理完，
    分钟数才计得准确；否则一次只出一个节点，会把同一分钟的感染当成分钟累加。

    为什么循环条件带 fresh > 0：没有新鲜橘子时无需再扩展，
    否则最后一层会白白多算一分钟。

复杂度：时间 O(m·n)（每个格子至多入队一次），空间 O(m·n)（队列）。
"""
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    minutes = 0
    while queue and fresh > 0:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc))
        minutes += 1

    return minutes if fresh == 0 else -1


if __name__ == "__main__":
    assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert oranges_rotting([[0, 2]]) == 0
    assert oranges_rotting([[1]]) == -1
    assert oranges_rotting([[2]]) == 0
    # 没有新鲜橘子，0 分钟
    assert oranges_rotting([[2, 2], [2, 2]]) == 0
    print("oranges_rotting: all tests passed")
