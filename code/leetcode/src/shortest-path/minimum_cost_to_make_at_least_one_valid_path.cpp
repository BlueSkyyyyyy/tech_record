// 1368. 使网格图至少有一条有效路径的最小代价
// 见 minimum_cost_to_make_at_least_one_valid_path.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <utility>
#include <vector>

int minCost(const std::vector<std::vector<int>> &grid) {
    int m = grid.size(), n = grid[0].size();
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, INF));
    dist[0][0] = 0;
    std::deque<std::pair<int, int>> dq;
    dq.push_back({0, 0});
    // 下标 0/1/2/3 分别对应路标 1/2/3/4：右、左、下、上
    const int dr[4] = {0, 0, 1, -1};
    const int dc[4] = {1, -1, 0, 0};
    while (!dq.empty()) {
        auto [r, c] = dq.front();
        dq.pop_front();
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                int cost = (grid[r][c] == i + 1) ? 0 : 1;
                int nd = dist[r][c] + cost;
                if (nd < dist[nr][nc]) {
                    dist[nr][nc] = nd;
                    if (cost == 0) {
                        dq.push_front({nr, nc});
                    } else {
                        dq.push_back({nr, nc});
                    }
                }
            }
        }
    }
    return dist[m - 1][n - 1];
}

int main() {
    std::vector<std::vector<int>> g1 = {
        {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}};
    assert(minCost(g1) == 3);

    std::vector<std::vector<int>> g2 = {{1, 1, 3}, {3, 2, 2}, {1, 1, 4}};
    assert(minCost(g2) == 0);

    std::vector<std::vector<int>> g3 = {{1, 2}, {4, 3}};
    assert(minCost(g3) == 1);

    std::vector<std::vector<int>> g4 = {{2, 2}, {2, 2}};
    assert(minCost(g4) == 2);

    std::cout << "minimum_cost_to_make_at_least_one_valid_path: all tests passed\n";
    return 0;
}
