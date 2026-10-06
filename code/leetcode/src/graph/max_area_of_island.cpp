// 695. 岛屿的最大面积
// 见 max_area_of_island.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int dfs(std::vector<std::vector<int>> &grid, int r, int c) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != 1) return 0;
    grid[r][c] = 0;
    return 1 + dfs(grid, r + 1, c) + dfs(grid, r - 1, c) + dfs(grid, r, c + 1) +
           dfs(grid, r, c - 1);
}

int maxAreaOfIsland(std::vector<std::vector<int>> grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int rows = grid.size();
    int cols = grid[0].size();
    int best = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 1) {
                best = std::max(best, dfs(grid, r, c));
            }
        }
    }
    return best;
}

int main() {
    std::vector<std::vector<int>> g1 = {
        {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0},
        {0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
    };
    assert(maxAreaOfIsland(g1) == 6);

    assert(maxAreaOfIsland({{0, 0, 0, 0}}) == 0);
    assert(maxAreaOfIsland({{1}}) == 1);
    assert(maxAreaOfIsland({{1, 1}, {1, 1}}) == 4);

    std::cout << "max_area_of_island: all tests passed\n";
    return 0;
}
