// 847. 访问所有节点的最短路径
// 见 shortest_path_visiting_all_nodes.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

int shortestPathLength(std::vector<std::vector<int>> &graph) {
    int n = static_cast<int>(graph.size());
    int full = (1 << n) - 1;

    std::vector<std::vector<int>> dist(1 << n, std::vector<int>(n, -1));
    std::queue<std::pair<int, int>> q;
    for (int i = 0; i < n; ++i) {
        int mask = 1 << i;
        dist[mask][i] = 0;
        q.push({mask, i});
    }

    while (!q.empty()) {
        auto [mask, u] = q.front();
        q.pop();
        int d = dist[mask][u];
        if (mask == full) {
            return d;
        }
        for (int v : graph[u]) {
            int nxt = mask | (1 << v);
            if (dist[nxt][v] == -1) {
                dist[nxt][v] = d + 1;
                q.push({nxt, v});
            }
        }
    }
    return -1;
}

int main() {
    {
        std::vector<std::vector<int>> g = {{1, 2, 3}, {0}, {0}, {0}};
        assert(shortestPathLength(g) == 4);
    }
    {
        std::vector<std::vector<int>> g = {{1}, {0, 2, 4}, {1, 3, 4}, {2}, {1, 2}};
        assert(shortestPathLength(g) == 4);
    }
    {
        std::vector<std::vector<int>> g = {{}};
        assert(shortestPathLength(g) == 0);
    }
    {
        std::vector<std::vector<int>> g = {{1}, {0}};
        assert(shortestPathLength(g) == 1);
    }

    std::cout << "shortest_path_visiting_all_nodes: all tests passed\n";
    return 0;
}
