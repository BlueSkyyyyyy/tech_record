// 787. K 站中转内最便宜的航班
// 见 cheapest_flights_within_k_stops.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int findCheapestPrice(int n, const std::vector<std::vector<int>> &flights, int src,
                      int dst, int k) {
    const long long INF = LLONG_MAX;
    std::vector<long long> dist(n, INF);
    dist[src] = 0;
    for (int round = 0; round <= k; ++round) {
        std::vector<long long> prev = dist;
        for (const auto &e : flights) {
            int u = e[0], v = e[1], w = e[2];
            if (prev[u] != INF && prev[u] + w < dist[v]) {
                dist[v] = prev[u] + w;
            }
        }
    }
    return dist[dst] == INF ? -1 : static_cast<int>(dist[dst]);
}

int main() {
    std::vector<std::vector<int>> f1 = {
        {0, 1, 100}, {1, 2, 100}, {2, 0, 100}, {1, 3, 600}, {2, 3, 200}};
    assert(findCheapestPrice(4, f1, 0, 3, 1) == 700);

    std::vector<std::vector<int>> f2 = {{0, 1, 100}, {1, 2, 100}, {0, 2, 500}};
    assert(findCheapestPrice(3, f2, 0, 2, 1) == 200);
    assert(findCheapestPrice(3, f2, 0, 2, 0) == 500);

    std::vector<std::vector<int>> f3 = {{1, 0, 100}};
    assert(findCheapestPrice(2, f3, 0, 1, 1) == -1);

    std::cout << "cheapest_flights_within_k_stops: all tests passed\n";
    return 0;
}
