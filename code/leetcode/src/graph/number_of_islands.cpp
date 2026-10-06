// 200. 岛屿数量
// 见 number_of_islands.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void dfs(std::vector<std::vector<char>> &grid, int r, int c) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') return;
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}

int numIslands(std::vector<std::vector<char>> &grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int rows = grid.size();
    int cols = grid[0].size();
    int count = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == '1') {
                ++count;
                dfs(grid, r, c);
            }
        }
    }
    return count;
}

int main() {
    std::vector<std::vector<char>> g1 = {
        {'1', '1', '1', '1', '0'},
        {'1', '1', '0', '1', '0'},
        {'1', '1', '0', '0', '0'},
        {'0', '0', '0', '0', '0'},
    };
    assert(numIslands(g1) == 1);

    std::vector<std::vector<char>> g2 = {
        {'1', '1', '0', '0', '0'},
        {'1', '1', '0', '0', '0'},
        {'0', '0', '1', '0', '0'},
        {'0', '0', '0', '1', '1'},
    };
    assert(numIslands(g2) == 3);

    std::vector<std::vector<char>> empty;
    assert(numIslands(empty) == 0);

    std::vector<std::vector<char>> zero = {{'0'}};
    assert(numIslands(zero) == 0);

    std::vector<std::vector<char>> one = {{'1'}};
    assert(numIslands(one) == 1);

    std::cout << "number_of_islands: all tests passed\n";
    return 0;
}
