// 883. 三维形体投影面积
// 见 projection_area_of_3d_shapes.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int projectionArea(std::vector<std::vector<int>>& grid) {
    int n = static_cast<int>(grid.size());
    int top = 0, front = 0, side = 0;
    for (int i = 0; i < n; ++i) {
        int rowMax = 0;
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] > 0) {
                ++top;
            }
            rowMax = std::max(rowMax, grid[i][j]);
        }
        front += rowMax;
    }
    for (int j = 0; j < n; ++j) {
        int colMax = 0;
        for (int i = 0; i < n; ++i) {
            colMax = std::max(colMax, grid[i][j]);
        }
        side += colMax;
    }
    return top + front + side;
}

int main() {
    std::vector<std::vector<int>> a{{2}};
    std::vector<std::vector<int>> b{{1, 2}, {3, 4}};
    std::vector<std::vector<int>> c{{1, 0}, {0, 2}};
    std::vector<std::vector<int>> d{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};

    assert(projectionArea(a) == 5);
    assert(projectionArea(b) == 17);
    assert(projectionArea(c) == 8);
    assert(projectionArea(d) == 14);

    std::cout << "projection_area_of_3d_shapes: all tests passed\n";
    return 0;
}
