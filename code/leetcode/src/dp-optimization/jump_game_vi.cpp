// 1696. 跳跃游戏 VI（DP + 单调队列）
// 见 jump_game_vi.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int maxResult(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    dp[0] = nums[0];
    std::deque<int> dq;
    dq.push_back(0);
    for (int i = 1; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) dq.pop_front();
        dp[i] = nums[i] + dp[dq.front()];
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return dp[n - 1];
}

int main() {
    assert(maxResult({1, -1, -2, 4, -7, 3}, 2) == 7);
    assert(maxResult({10, -5, -2, 4, 0, 3}, 3) == 17);
    assert(maxResult({1, -5, -20, 4, -1, 3, -6, -3}, 2) == 0);
    assert(maxResult({5}, 1) == 5);
    std::cout << "jump_game_vi: all tests passed\n";
    return 0;
}
