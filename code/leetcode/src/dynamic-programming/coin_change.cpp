// 322. 零钱兑换
// 见 coin_change.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int coinChange(const std::vector<int> &coins, int amount) {
    const int INF = amount + 1;
    std::vector<int> dp(amount + 1, INF);
    dp[0] = 0;
    for (int coin : coins) {
        for (int i = coin; i <= amount; ++i) {
            dp[i] = std::min(dp[i], dp[i - coin] + 1);
        }
    }
    return dp[amount] == INF ? -1 : dp[amount];
}

int main() {
    std::vector<int> a = {1, 2, 5};
    std::vector<int> b = {2};
    std::vector<int> c = {1};
    std::vector<int> d = {1};
    std::vector<int> e = {2, 5, 10, 1};
    assert(coinChange(a, 11) == 3);
    assert(coinChange(b, 3) == -1);
    assert(coinChange(c, 0) == 0);
    assert(coinChange(d, 2) == 2);
    assert(coinChange(e, 27) == 4);
    std::cout << "coin_change: all tests passed\n";
    return 0;
}
