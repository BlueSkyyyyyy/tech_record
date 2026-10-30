// 1751. 最多可以参加的会议数目 II（排序 + 二分 + DP）
// 见 maximum_number_of_events_that_can_be_attended_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxValue(std::vector<std::vector<int>> events, int k) {
    std::sort(events.begin(), events.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) { return a[1] < b[1]; });
    int n = static_cast<int>(events.size());
    std::vector<int> ends(n);
    for (int i = 0; i < n; ++i) ends[i] = events[i][1];

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(k + 1, 0));
    for (int i = 1; i <= n; ++i) {
        int start = events[i - 1][0];
        int value = events[i - 1][2];
        int p = static_cast<int>(std::lower_bound(ends.begin(), ends.end(), start) - ends.begin());
        for (int j = 1; j <= k; ++j) {
            dp[i][j] = std::max(dp[i - 1][j], dp[p][j - 1] + value);
        }
    }
    return dp[n][k];
}

int main() {
    assert(maxValue({{1, 2, 4}, {3, 4, 3}, {2, 3, 1}}, 2) == 7);
    assert(maxValue({{1, 2, 4}, {3, 4, 3}, {2, 3, 10}}, 2) == 10);
    assert(maxValue({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}}, 3) == 9);
    assert(maxValue({{1, 3, 5}}, 1) == 5);
    std::cout << "maximum_number_of_events_that_can_be_attended_ii: all tests passed\n";
    return 0;
}
