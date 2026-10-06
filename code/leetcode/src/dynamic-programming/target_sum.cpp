// 494. 目标和
// 见 target_sum.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

int findTargetSumWays(const std::vector<int> &nums, int target) {
    int total = 0;
    for (int x : nums) total += x;
    if (std::abs(target) > total || (total + target) % 2 != 0) return 0;
    int p = (total + target) / 2;
    std::vector<int> dp(p + 1, 0);
    dp[0] = 1;
    for (int num : nums) {
        for (int i = p; i >= num; --i) {
            dp[i] += dp[i - num];
        }
    }
    return dp[p];
}

int main() {
    std::vector<int> a = {1, 1, 1, 1, 1};
    std::vector<int> b = {1};
    std::vector<int> c = {1, 2, 3};
    std::vector<int> d = {1, 2};
    std::vector<int> e = {1, 0};
    std::vector<int> f = {1, 2};
    assert(findTargetSumWays(a, 3) == 5);
    assert(findTargetSumWays(b, 1) == 1);
    assert(findTargetSumWays(c, 0) == 2);
    assert(findTargetSumWays(d, 3) == 1);
    assert(findTargetSumWays(e, 1) == 2);
    assert(findTargetSumWays(f, 5) == 0);
    std::cout << "target_sum: all tests passed\n";
    return 0;
}
