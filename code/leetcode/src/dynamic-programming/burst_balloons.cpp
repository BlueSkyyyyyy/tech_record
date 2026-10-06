// 312. 戳气球
// 见 burst_balloons.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int maxCoins(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<int> vals(n + 2, 1);
    for (int i = 0; i < n; ++i) vals[i + 1] = nums[i];
    int total = n + 2;
    std::vector<std::vector<int>> dp(total, std::vector<int>(total, 0));
    for (int length = 2; length < total; ++length) {
        for (int i = 0; i + length < total; ++i) {
            int j = i + length;
            for (int k = i + 1; k < j; ++k) {
                int gain = vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j];
                dp[i][j] = std::max(dp[i][j], gain);
            }
        }
    }
    return dp[0][total - 1];
}

int main() {
    assert(maxCoins(std::vector<int>{3, 1, 5, 8}) == 167);
    assert(maxCoins(std::vector<int>{1, 5}) == 10);
    assert(maxCoins(std::vector<int>{1}) == 1);
    assert(maxCoins(std::vector<int>{}) == 0);
    std::cout << "burst_balloons: all tests passed\n";
    return 0;
}
