// 1235. 规划兼职工作（DP + 二分查找）
// 见 job_scheduling.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int jobScheduling(const std::vector<int> &startTime, const std::vector<int> &endTime,
                  const std::vector<int> &profit) {
    int n = static_cast<int>(startTime.size());
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return endTime[a] < endTime[b]; });

    std::vector<int> ends(n);
    for (int i = 0; i < n; ++i) ends[i] = endTime[idx[i]];

    std::vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        int j = idx[i - 1];
        int s = startTime[j], p = profit[j];
        // ends[0..i-2] 中 <= s 的个数 = 可接的最后一个工作下标 + 1
        int m = static_cast<int>(std::upper_bound(ends.begin(), ends.begin() + (i - 1), s) -
                                 ends.begin());
        dp[i] = std::max(dp[i - 1], dp[m] + p);
    }
    return dp[n];
}

int main() {
    assert(jobScheduling({1, 2, 3, 3}, {3, 4, 5, 6}, {50, 10, 40, 70}) == 120);
    assert(jobScheduling({1, 2, 3, 4, 6}, {3, 5, 10, 6, 9}, {20, 20, 100, 70, 60}) == 150);
    assert(jobScheduling({1, 1, 1}, {2, 3, 4}, {5, 6, 7}) == 7);
    std::cout << "job_scheduling: all tests passed\n";
    return 0;
}
