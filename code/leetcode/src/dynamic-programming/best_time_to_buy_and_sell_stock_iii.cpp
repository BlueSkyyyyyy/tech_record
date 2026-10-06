// 123. 买卖股票的最佳时机 III
// 见 best_time_to_buy_and_sell_stock_iii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfitIII(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long buy1 = NEG, sell1 = 0, buy2 = NEG, sell2 = 0;
    for (int p : prices) {
        buy1 = std::max(buy1, -static_cast<long long>(p));
        sell1 = std::max(sell1, buy1 + p);
        buy2 = std::max(buy2, sell1 - p);
        sell2 = std::max(sell2, buy2 + p);
    }
    return static_cast<int>(sell2);
}

int main() {
    std::vector<int> a = {3, 3, 5, 0, 0, 3, 1, 4};
    std::vector<int> b = {1, 2, 3, 4, 5};
    std::vector<int> c = {7, 6, 4, 3, 1};
    std::vector<int> d = {1};
    std::vector<int> e = {2, 1, 2, 0, 1};
    assert(maxProfitIII(a) == 6);
    assert(maxProfitIII(b) == 4);
    assert(maxProfitIII(c) == 0);
    assert(maxProfitIII(d) == 0);
    assert(maxProfitIII(e) == 2);
    std::cout << "best_time_to_buy_and_sell_stock_iii: all tests passed\n";
    return 0;
}
