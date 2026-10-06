"""200. 岛屿数量（Number of Islands）

题目：给你一个由字符 '1'（陆地）和 '0'（水）组成的二维网格 grid，
请计算网格中岛屿的数量。岛屿被水包围，并且通过水平或垂直方向相邻的陆地连接而成。
你可以假设网格的四条边均被水包围。

思路（网格 DFS，一次淹没一整座岛）：
    把每个陆地格子看作图中的一个节点，上下左右相邻的陆地之间有边。
    于是「数岛屿」就是「数连通块的个数」：
    从左到右、从上到下扫描，遇到一个还没访问过的 '1'，说明发现了一座新岛，
    计数加一并从它出发做一次 DFS，把整座岛上的 '1' 全部改成 '0'（淹没），
    这样后续扫描就不会重复计数。

    为什么直接改 grid 而不用额外的 visited 表：访问过的陆地不会再有用，
    就地涂成 '0' 既当「已访问」标记又省了 O(mn) 的额外空间。

    为什么 DFS 的边界判断要写在最前面：越界或踩到水就返回，
    这是递归的终止条件，也是防止无限递归的关键。

复杂度：时间 O(m·n)（每个格子至多被访问一次），空间 O(m·n)（最坏全是陆地，
    递归栈深度可达 m·n）。
"""


def num_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count


if __name__ == "__main__":
    g1 = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]
    assert num_islands(g1) == 1

    g2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert num_islands(g2) == 3

    assert num_islands([]) == 0
    assert num_islands([["0"]]) == 0
    assert num_islands([["1"]]) == 1
    print("num_islands: all tests passed")
