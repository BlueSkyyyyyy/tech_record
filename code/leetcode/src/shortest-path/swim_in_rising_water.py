"""778. 水位上升的泳池中游泳（Swim in Rising Water）

题目：给一个 n×n 的方格，grid[i][j] 是平台高度（是 0~n²-1 的一个排列）。时刻 t 时
水位为 t，只有高度不超过 t 的平台能被游到。你可以在水位足够时，在相邻（上下左右）
平台间瞬移。求从 (0,0) 游到 (n-1,n-1) 的最早时刻。

思路（二分答案 + BFS 判定）：
    「最早时刻」本身不好直接求，但「在时刻 t 能否到达终点」很好判定：只保留高度
    ≤ t 的格子，做一次 BFS 看起终点连通与否。而且这个判定随 t 单调——t 越大可用
    格子越多，连通性只增不减。于是可以二分最小的可行 t。

    答案的下界是 max(grid[0][0], grid[n-1][n-1])（起点终点本身必须能被淹没），
    上界是 n²-1（最大可能高度）。二分这个区间，每次用 BFS 判定。

    为什么二分可行：单调性是二分的命根子。本题「能到达」关于 t 单调成立，所以
    「可行 / 不可行」在数轴上呈现为一段「否」接一段「是」，二分能精准找到交界。

复杂度：时间 O(n²·log n)（二分 log(n²) 轮，每轮 BFS O(n²)），空间 O(n²)。
"""
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def swim_in_water(grid):
    n = len(grid)

    def can_reach(t):
        if grid[0][0] > t or grid[n - 1][n - 1] > t:
            return False
        seen = [[False] * n for _ in range(n)]
        seen[0][0] = True
        queue = deque([(0, 0)])
        while queue:
            r, c = queue.popleft()
            if r == n - 1 and c == n - 1:
                return True
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and not seen[nr][nc]
                    and grid[nr][nc] <= t
                ):
                    seen[nr][nc] = True
                    queue.append((nr, nc))
        return False

    lo = max(grid[0][0], grid[n - 1][n - 1])
    hi = n * n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if can_reach(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


if __name__ == "__main__":
    assert swim_in_water([[0, 2], [1, 3]]) == 3
    assert (
        swim_in_water(
            [
                [0, 1, 2, 3, 4],
                [24, 23, 22, 21, 5],
                [12, 13, 14, 15, 16],
                [11, 17, 18, 19, 20],
                [10, 9, 8, 7, 6],
            ]
        )
        == 16
    )
    # 只有一格，起点即终点
    assert swim_in_water([[0]]) == 0
    print("swim_in_water: all tests passed")
