// 1334. 阈值距离内邻居最少的城市
// 见 find_the_city_with_the_smallest_number_of_neighbors.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int findTheCity(int n, const std::vector<std::vector<int>> &edges,
                int distanceThreshold) {
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, INF));
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
    }
    for (const auto &e : edges) {
        int u = e[0], v = e[1], w = e[2];
        dist[u][v] = std::min(dist[u][v], w);
        dist[v][u] = std::min(dist[v][u], w);
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (dist[i][k] == INF) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    int bestCity = -1, bestCount = n + 1;
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j && dist[i][j] <= distanceThreshold) {
                ++cnt;
            }
        }
        if (cnt <= bestCount) {
            bestCount = cnt;
            bestCity = i;
        }
    }
    return bestCity;
}

int main() {
    assert(findTheCity(4, {{0, 1, 3}, {1, 2, 1}, {1, 3, 4}, {2, 3, 1}}, 4) == 3);
    assert(findTheCity(5,
                       {{0, 1, 2}, {0, 4, 8}, {1, 2, 3}, {1, 4, 2}, {2, 3, 1}, {3, 4, 1}},
                       2) == 0);

    std::cout << "find_the_city_with_the_smallest_number_of_neighbors: all tests passed\n";
    return 0;
}
