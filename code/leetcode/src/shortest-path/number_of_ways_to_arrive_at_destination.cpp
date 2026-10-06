// 1976. 到达目的地的方案数
// 见 number_of_ways_to_arrive_at_destination.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <functional>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

const long long MOD = 1000000007LL;

int countPaths(int n, const std::vector<std::vector<int>> &roads) {
    std::vector<std::vector<std::pair<int, int>>> graph(n);
    for (const auto &e : roads) {
        graph[e[0]].push_back({e[1], e[2]});
        graph[e[1]].push_back({e[0], e[2]});
    }
    std::vector<long long> dist(n, LLONG_MAX), ways(n, 0);
    using P = std::pair<long long, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    dist[0] = 0;
    ways[0] = 1;
    heap.push({0, 0});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) {
            continue;
        }
        for (auto [v, w] : graph[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                ways[v] = ways[u];
                heap.push({nd, v});
            } else if (nd == dist[v]) {
                ways[v] = (ways[v] + ways[u]) % MOD;
            }
        }
    }
    return static_cast<int>(ways[n - 1] % MOD);
}

int main() {
    assert(countPaths(4, {{0, 1, 1}, {0, 2, 1}, {1, 3, 1}, {2, 3, 1}}) == 2);
    assert(countPaths(3, {{0, 1, 1}, {1, 2, 1}, {0, 2, 5}}) == 1);
    assert(countPaths(2, {{0, 1, 1}}) == 1);
    assert(countPaths(4, {{0, 3, 2}, {0, 1, 1}, {1, 3, 1}, {0, 2, 1}, {2, 3, 1}}) == 3);

    std::cout << "number_of_ways_to_arrive_at_destination: all tests passed\n";
    return 0;
}
