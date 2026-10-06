// 778. 水位上升的泳池中游泳
// 见 swim_in_rising_water.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

bool canReach(const std::vector<std::vector<int>> &grid, int t) {
    int n = grid.size();
    if (grid[0][0] > t || grid[n - 1][n - 1] > t) {
        return false;
    }
    std::vector<std::vector<bool>> seen(n, std::vector<bool>(n, false));
    seen[0][0] = true;
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        if (r == n - 1 && c == n - 1) {
            return true;
        }
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && !seen[nr][nc] &&
                grid[nr][nc] <= t) {
                seen[nr][nc] = true;
                q.push({nr, nc});
            }
        }
    }
    return false;
}

int swimInWater(const std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    int lo = std::max(grid[0][0], grid[n - 1][n - 1]);
    int hi = n * n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (canReach(grid, mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

int main() {
    std::vector<std::vector<int>> g1 = {{0, 2}, {1, 3}};
    assert(swimInWater(g1) == 3);

    std::vector<std::vector<int>> g2 = {
        {0, 1, 2, 3, 4},
        {24, 23, 22, 21, 5},
        {12, 13, 14, 15, 16},
        {11, 17, 18, 19, 20},
        {10, 9, 8, 7, 6},
    };
    assert(swimInWater(g2) == 16);

    std::vector<std::vector<int>> g3 = {{0}};
    assert(swimInWater(g3) == 0);

    std::cout << "swim_in_rising_water: all tests passed\n";
    return 0;
}
