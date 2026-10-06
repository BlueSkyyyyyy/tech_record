// 1631. 最小体力消耗路径
// 见 minimum_effort_path.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

int minimumEffortPath(const std::vector<std::vector<int>> &heights) {
    int m = heights.size(), n = heights[0].size();
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, INF));
    dist[0][0] = 0;
    using T = std::tuple<int, int, int>;
    std::priority_queue<T, std::vector<T>, std::greater<T>> heap;
    heap.push({0, 0, 0});
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!heap.empty()) {
        auto [effort, r, c] = heap.top();
        heap.pop();
        if (r == m - 1 && c == n - 1) {
            return effort;
        }
        if (effort > dist[r][c]) {
            continue;
        }
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                int ne = std::max(effort, std::abs(heights[nr][nc] - heights[r][c]));
                if (ne < dist[nr][nc]) {
                    dist[nr][nc] = ne;
                    heap.push({ne, nr, nc});
                }
            }
        }
    }
    return dist[m - 1][n - 1];
}

int main() {
    std::vector<std::vector<int>> h1 = {{1, 2, 2}, {3, 8, 2}, {5, 3, 5}};
    assert(minimumEffortPath(h1) == 2);

    std::vector<std::vector<int>> h2 = {{1, 2, 3}, {3, 8, 4}, {5, 3, 5}};
    assert(minimumEffortPath(h2) == 1);

    std::vector<std::vector<int>> h3 = {
        {1, 2, 1, 1, 1},
        {1, 2, 1, 2, 1},
        {1, 2, 1, 2, 1},
        {1, 2, 1, 2, 1},
        {1, 1, 1, 2, 1},
    };
    assert(minimumEffortPath(h3) == 0);

    std::cout << "minimum_effort_path: all tests passed\n";
    return 0;
}
