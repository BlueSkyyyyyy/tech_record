// 1696. 跳跃游戏 VI
// 见 jump_game_vi.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

int maxResult(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    dp[0] = nums[0];
    std::deque<int> dq;
    dq.push_back(0);
    for (int i = 1; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) {
            dq.pop_front();
        }
        dp[i] = nums[i] + dp[dq.front()];
        while (!dq.empty() && dp[dq.back()] <= dp[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
    }
    return dp[n - 1];
}

int main() {
    std::vector<int> a{1, -1, -2, 4, -7, 3};
    std::vector<int> b{10, -5, -2, 4, 0, 3};
    std::vector<int> c{1, -5, -20, 4, -1, 3, -6, -3};
    std::vector<int> d{0};
    std::vector<int> e{100, -100, -300, -300, -300, -100, 100};

    assert(maxResult(a, 2) == 7);
    assert(maxResult(b, 3) == 17);
    assert(maxResult(c, 2) == 0);
    assert(maxResult(d, 1) == 0);
    assert(maxResult(e, 4) == 0);
    std::cout << "max_result: all tests passed\n";
    return 0;
}
