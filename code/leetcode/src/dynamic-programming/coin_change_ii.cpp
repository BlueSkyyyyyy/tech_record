// 518. 零钱兑换 II
// 见 coin_change_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int change(int amount, const std::vector<int> &coins) {
    std::vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int coin : coins) {
        for (int i = coin; i <= amount; ++i) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}

int main() {
    std::vector<int> a = {1, 2, 5};
    std::vector<int> b = {2};
    std::vector<int> c = {10};
    std::vector<int> d = {};
    std::vector<int> e = {1, 2, 5};
    assert(change(5, a) == 4);
    assert(change(3, b) == 0);
    assert(change(10, c) == 1);
    assert(change(0, d) == 1);
    assert(change(7, e) == 6);
    std::cout << "coin_change_ii: all tests passed\n";
    return 0;
}
