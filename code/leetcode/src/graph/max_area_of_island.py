"""695. 岛屿的最大面积（Max Area of Island）

题目：给你一个大小为 m x n 的二进制矩阵 grid，1 表示陆地、0 表示水。
岛屿是四方向相连的 1 组成的连通块。返回网格中岛屿的最大面积；没有岛屿则返回 0。

思路（网格 DFS，让 DFS 返回面积）：
    和 200 岛屿数量同一个扫描框架：遇到未访问的 1 就落入一座新岛。
    区别在于这次不满足于「数岛」，而是想知道这座岛有多大——那就让 DFS 直接返回答案：
    站在一个陆地上，面积 = 1（自己）+ 上下左右四个方向能扩展出的面积之和。
    踩到越界或水时返回 0，作为加法的单位元。

    为什么用「返回值汇总」而不是全局变量：每个格子的贡献都是「自己 1 份」，
    由递归把四个方向的结果加起来往上返回，天然就把整座岛的面积汇总到入口，
    逻辑干净、也不依赖遍历顺序。这就是树题里「后序递归返回值」在网格上的版本。

复杂度：时间 O(m·n)（每个格子至多访问一次），空间 O(m·n)（递归栈最坏情形）。
"""


def max_area_of_island(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    best = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                best = max(best, dfs(r, c))
    return best


if __name__ == "__main__":
    g1 = [
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ]
    assert max_area_of_island(g1) == 6

    assert max_area_of_island([[0, 0, 0, 0, 0, 0, 0, 0]]) == 0
    assert max_area_of_island([[1]]) == 1
    assert max_area_of_island([[1, 1], [1, 1]]) == 4
    print("max_area_of_island: all tests passed")
