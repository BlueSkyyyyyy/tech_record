// 743. 网络延迟时间
// 见 network_delay_time.py 的题目与思路说明。
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

int networkDelayTime(const std::vector<std::vector<int>> &times, int n, int k) {
    std::vector<std::vector<std::pair<int, int>>> graph(n + 1);
    for (const auto &e : times) {
        graph[e[0]].push_back({e[1], e[2]});
    }
    const int INF = 1e9;
    std::vector<int> dist(n + 1, INF);
    dist[k] = 0;
    using P = std::pair<int, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    heap.push({0, k});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) {
            continue;
        }
        for (auto [v, w] : graph[u]) {
            int nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap.push({nd, v});
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; ++i) {
        if (dist[i] == INF) {
            return -1;
        }
        ans = std::max(ans, dist[i]);
    }
    return ans;
}

int main() {
    assert(networkDelayTime({{2, 1, 1}, {2, 3, 1}, {3, 4, 1}}, 4, 2) == 2);
    assert(networkDelayTime({{1, 2, 1}}, 2, 2) == -1);
    assert(networkDelayTime({{1, 2, 1}}, 2, 1) == 1);
    assert(networkDelayTime({{1, 2, 1}, {2, 3, 1}, {1, 3, 5}}, 3, 1) == 2);

    std::cout << "network_delay_time: all tests passed\n";
    return 0;
}
