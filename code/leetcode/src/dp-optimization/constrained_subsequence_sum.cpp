// 1425. 带限制的子序列和（DP + 单调队列）
// 见 constrained_subsequence_sum.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int constrainedSubsetSum(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    std::deque<int> dq;
    int best = nums[0];
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) dq.pop_front();
        int prev = dq.empty() ? 0 : std::max(0, dp[dq.front()]);
        dp[i] = nums[i] + prev;
        best = std::max(best, dp[i]);
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return best;
}

int main() {
    assert(constrainedSubsetSum({10, 2, -10, 5, 20}, 2) == 37);
    assert(constrainedSubsetSum({-1, -2, -3}, 1) == -1);
    assert(constrainedSubsetSum({10, -2, -10, -5, 20}, 2) == 23);
    assert(constrainedSubsetSum({-3}, 1) == -3);
    std::cout << "constrained_subsequence_sum: all tests passed\n";
    return 0;
}
