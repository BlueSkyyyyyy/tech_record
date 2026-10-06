// 892. 三维形体的表面积
// 见 surface_area_of_3d_shapes.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int surfaceArea(std::vector<std::vector<int>>& grid) {
    int n = static_cast<int>(grid.size());
    int total = 0;
    int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int v = grid[i][j];
            if (v == 0) {
                continue;
            }
            total += 2;
            for (auto& d : dirs) {
                int ni = i + d[0];
                int nj = j + d[1];
                int neighbor =
                    (ni >= 0 && ni < n && nj >= 0 && nj < n) ? grid[ni][nj] : 0;
                if (v > neighbor) {
                    total += v - neighbor;
                }
            }
        }
    }
    return total;
}

int main() {
    std::vector<std::vector<int>> a{{2}};
    std::vector<std::vector<int>> b{{1, 2}, {3, 4}};
    std::vector<std::vector<int>> c{{1, 0}, {0, 2}};
    std::vector<std::vector<int>> d{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};

    assert(surfaceArea(a) == 10);
    assert(surfaceArea(b) == 34);
    assert(surfaceArea(c) == 16);
    assert(surfaceArea(d) == 32);

    std::cout << "surface_area_of_3d_shapes: all tests passed\n";
    return 0;
}
