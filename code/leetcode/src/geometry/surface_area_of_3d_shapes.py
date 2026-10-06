"""892. 三维形体的表面积（Surface Area of 3D Shapes）

题目：给定 n×n 的格子 grid，grid[i][j] 表示在位置 (i, j) 上堆叠的正方体
个数。求整个立体图形的表面积（贴合的面不算）。

思路（逐个方块数「露出来的面」）：
    对每个有 v = grid[i][j] 根方块的格子：
    - 上下两个底面永远露出来，贡献 2；
    - 朝四个方向（前后左右）的侧面，每个方向露出的高度是
      max(0, v - 邻居高度)：与相邻柱子贴住的部分被挡住，高出的部分露出来。
    把四个方向各自贡献的高度加起来即可，不需要构造三维模型。

    也可以先按 4*v + 2 计，再对每条相邻边减去 2*min(v, 邻居)，效果相同；
    这里采用「邻居缺多少就差多少」的写法，更直观。

复杂度：时间 O(n^2)（每个格子看四个方向），空间 O(1)。
"""


def surface_area(grid):
    n = len(grid)
    total = 0
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v == 0:
                continue
            total += 2
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                neighbor = grid[ni][nj] if 0 <= ni < n and 0 <= nj < n else 0
                if v > neighbor:
                    total += v - neighbor
    return total


if __name__ == "__main__":
    assert surface_area([[2]]) == 10
    assert surface_area([[1, 2], [3, 4]]) == 34
    assert surface_area([[1, 0], [0, 2]]) == 16
    assert surface_area([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == 32
    print("surface_area_of_3d_shapes: all tests passed")
