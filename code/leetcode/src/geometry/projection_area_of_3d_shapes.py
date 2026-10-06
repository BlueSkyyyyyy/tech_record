"""883. 三维形体投影面积（Projection Area of 3D Shapes）

题目：给定 n×n 的格子 grid，grid[i][j] 是堆叠的方块数。求这个立体在三个
坐标平面上的**投影面积之和**。

思路（三个方向各自一个统计量）：
    - 俯视 xy 平面：只要格子上有方块（>0）就投影出一个单位面积，
      所以是「非零格子个数」；
    - 正视 xz 平面（沿列方向看）：每一行看到的高度是该行的最大值，
      把每行最大值相加；
    - 侧视 yz 平面（沿行方向看）：每一列看到的高度是该列的最大值，
      把每列最大值相加。
    三者之和就是答案。因为是投影，重叠的方块只算一次，所以取的是最大值而
    不是求和。

复杂度：时间 O(n^2)，空间 O(1)。
"""


def projection_area(grid):
    n = len(grid)
    top = sum(1 for row in grid for v in row if v > 0)
    front = sum(max(row) for row in grid)
    side = sum(max(grid[i][j] for i in range(n)) for j in range(n))
    return top + front + side


if __name__ == "__main__":
    assert projection_area([[2]]) == 5
    assert projection_area([[1, 2], [3, 4]]) == 17
    assert projection_area([[1, 0], [0, 2]]) == 8
    assert projection_area([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == 14
    print("projection_area_of_3d_shapes: all tests passed")
