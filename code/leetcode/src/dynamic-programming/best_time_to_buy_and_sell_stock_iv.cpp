// 188. 买卖股票的最佳时机 IV
// 见 best_time_to_buy_and_sell_stock_iv.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfitIV(int k, const std::vector<int> &prices) {
    if (k <= 0 || prices.empty()) return 0;
    const long long NEG = -1e18;
    std::vector<long long> buy(k + 1, NEG), sell(k + 1, 0);
    for (int p : prices) {
        for (int j = 1; j <= k; ++j) {
            buy[j] = std::max(buy[j], sell[j - 1] - static_cast<long long>(p));
            sell[j] = std::max(sell[j], buy[j] + p);
        }
    }
    return static_cast<int>(sell[k]);
}

int main() {
    std::vector<int> a = {2, 4, 1};
    std::vector<int> b = {3, 2, 6, 5, 0, 3};
    std::vector<int> c = {1, 2, 3};
    std::vector<int> d = {7, 1, 5, 3, 6, 4};
    std::vector<int> e = {1, 2, 3, 4, 5};
    std::vector<int> f = {};
    assert(maxProfitIV(2, a) == 2);
    assert(maxProfitIV(2, b) == 7);
    assert(maxProfitIV(0, c) == 0);
    assert(maxProfitIV(1, d) == 5);
    assert(maxProfitIV(100, e) == 4);
    assert(maxProfitIV(2, f) == 0);
    std::cout << "best_time_to_buy_and_sell_stock_iv: all tests passed\n";
    return 0;
}
