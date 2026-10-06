// 1928. 规定时间内到达终点的最小花费
// 见 minimum_cost_to_reach_destination_in_time.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

int minCost(int maxTime, const std::vector<std::vector<int>> &edges,
            const std::vector<int> &passingFees) {
    int n = passingFees.size();
    std::vector<std::vector<std::pair<int, int>>> graph(n);
    for (const auto &e : edges) {
        graph[e[0]].push_back({e[1], e[2]});
        graph[e[1]].push_back({e[0], e[2]});
    }
    const int INF = 1e9;
    std::vector<std::vector<int>> dp(maxTime + 1, std::vector<int>(n, INF));
    dp[0][0] = passingFees[0];
    for (int t = 0; t <= maxTime; ++t) {
        for (int u = 0; u < n; ++u) {
            if (dp[t][u] == INF) {
                continue;
            }
            for (auto [v, w] : graph[u]) {
                int nt = t + w;
                if (nt <= maxTime && dp[t][u] + passingFees[v] < dp[nt][v]) {
                    dp[nt][v] = dp[t][u] + passingFees[v];
                }
            }
        }
    }
    int ans = INF;
    for (int t = 0; t <= maxTime; ++t) {
        if (dp[t][n - 1] < ans) {
            ans = dp[t][n - 1];
        }
    }
    return ans == INF ? -1 : ans;
}

int main() {
    assert(minCost(30, {{0, 1, 10}, {1, 2, 10}, {0, 2, 30}}, {5, 1, 2}) == 7);
    assert(minCost(5, {{0, 1, 5}}, {1, 3}) == 4);
    assert(minCost(4, {{0, 1, 5}}, {1, 3}) == -1);

    std::cout << "minimum_cost_to_reach_destination_in_time: all tests passed\n";
    return 0;
}
