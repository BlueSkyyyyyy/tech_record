"""733. 图像渲染（Flood Fill）

题目：有一幅以二维整数数组表示的图画 image，image[i][j] 表示该位置像素的颜色。
给你三个整数 sr、sc、color，表示从坐标 (sr, sc) 开始渲染。
将与起始像素颜色相同且四方向相连的像素都染成 color，返回渲染后的图像。

思路（网格 DFS，漫水填充）：
    本题和「岛屿数量」是同一个连通块问题，只是目标从「计数」变成「染色」：
    从 (sr, sc) 出发，凡是与它四方向相连、且颜色等于起始颜色的格子，一律改成新颜色。

    两个关键细节：
    1. 先把起始颜色记为 start。DFS 只沿着「颜色仍等于 start」的格子扩散；
       一旦某个格子被改成新颜色，它就不再满足条件，天然充当了「已访问」标记。
    2. 若 start == color，扩散条件永远成立、改色又不产生变化，会陷入死循环。
       所以开头直接返回原图。

    为什么用 DFS 而不是 BFS：两者都能做，DFS 代码更短；若担心递归过深，
    可换成显式栈的迭代写法。

复杂度：时间 O(m·n)（每个格子至多处理一次），空间 O(m·n)（递归栈最坏情形）。
"""


def flood_fill(image, sr, sc, color):
    rows, cols = len(image), len(image[0])
    start = image[sr][sc]
    if start == color:
        return image

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or image[r][c] != start:
            return
        image[r][c] = color
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return image


if __name__ == "__main__":
    assert flood_fill([[1, 1, 1], [1, 1, 0], [1, 0, 1]], 1, 1, 2) == [
        [2, 2, 2],
        [2, 2, 0],
        [2, 0, 1],
    ]
    assert flood_fill([[0, 0, 0], [0, 0, 0]], 0, 0, 0) == [[0, 0, 0], [0, 0, 0]]
    assert flood_fill([[0, 0, 0], [0, 1, 1]], 1, 1, 1) == [[0, 0, 0], [0, 1, 1]]
    assert flood_fill([[5]], 0, 0, 9) == [[9]]
    print("flood_fill: all tests passed")
