// 994. 腐烂的橘子
// 见 rotting_oranges.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

int orangesRotting(std::vector<std::vector<int>> grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    std::queue<std::pair<int, int>> q;
    int fresh = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 2) {
                q.push({r, c});
            } else if (grid[r][c] == 1) {
                ++fresh;
            }
        }
    }

    int minutes = 0;
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty() && fresh > 0) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            auto [r, c] = q.front();
            q.pop();
            for (int d = 0; d < 4; ++d) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                    grid[nr][nc] == 1) {
                    grid[nr][nc] = 2;
                    --fresh;
                    q.push({nr, nc});
                }
            }
        }
        ++minutes;
    }
    return fresh == 0 ? minutes : -1;
}

int main() {
    std::vector<std::vector<int>> g1 = {{2, 1, 1}, {1, 1, 0}, {0, 1, 1}};
    assert(orangesRotting(g1) == 4);

    std::vector<std::vector<int>> g2 = {{2, 1, 1}, {0, 1, 1}, {1, 0, 1}};
    assert(orangesRotting(g2) == -1);

    std::vector<std::vector<int>> g3 = {{0, 2}};
    assert(orangesRotting(g3) == 0);

    std::vector<std::vector<int>> g4 = {{1}};
    assert(orangesRotting(g4) == -1);

    std::vector<std::vector<int>> g5 = {{2, 2}, {2, 2}};
    assert(orangesRotting(g5) == 0);

    std::cout << "rotting_oranges: all tests passed\n";
    return 0;
}
