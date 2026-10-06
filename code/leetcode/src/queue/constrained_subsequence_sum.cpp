// 1425. 带限制的子序列和
// 见 constrained_subsequence_sum.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int constrainedSubsetSum(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    std::deque<int> dq;
    int ans = nums[0];
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) {
            dq.pop_front();
        }
        int best = dq.empty() ? 0 : std::max(0, dp[dq.front()]);
        dp[i] = nums[i] + best;
        while (!dq.empty() && dp[dq.back()] <= dp[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
        ans = std::max(ans, dp[i]);
    }
    return ans;
}

int main() {
    std::vector<int> a{10, 2, -10, 5, 20};
    std::vector<int> b{-1, -2, -3};
    std::vector<int> c{10, -2, -10, -5, 20};
    std::vector<int> d{-5};
    std::vector<int> e{1, 2, 3, 4, 5};

    assert(constrainedSubsetSum(a, 2) == 37);
    assert(constrainedSubsetSum(b, 1) == -1);
    assert(constrainedSubsetSum(c, 2) == 23);
    assert(constrainedSubsetSum(d, 1) == -5);
    assert(constrainedSubsetSum(e, 2) == 15);
    std::cout << "constrained_subset_sum: all tests passed\n";
    return 0;
}
