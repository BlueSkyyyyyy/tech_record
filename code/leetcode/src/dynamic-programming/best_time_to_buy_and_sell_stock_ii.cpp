// 122. 买卖股票的最佳时机 II
// 见 best_time_to_buy_and_sell_stock_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfitII(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, cash - static_cast<long long>(p));
        cash = std::max(cash, hold + p);
    }
    return static_cast<int>(cash);
}

int main() {
    std::vector<int> a = {7, 1, 5, 3, 6, 4};
    std::vector<int> b = {7, 6, 4, 3, 1};
    std::vector<int> c = {1, 2, 3, 4, 5};
    std::vector<int> d = {1};
    std::vector<int> e = {2, 4, 1};
    assert(maxProfitII(a) == 7);
    assert(maxProfitII(b) == 0);
    assert(maxProfitII(c) == 4);
    assert(maxProfitII(d) == 0);
    assert(maxProfitII(e) == 2);
    std::cout << "best_time_to_buy_and_sell_stock_ii: all tests passed\n";
    return 0;
}
