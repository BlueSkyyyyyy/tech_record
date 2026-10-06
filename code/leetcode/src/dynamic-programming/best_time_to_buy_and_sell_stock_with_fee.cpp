// 714. 买卖股票的最佳时机含手续费
// 见 best_time_to_buy_and_sell_stock_with_fee.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfitFee(const std::vector<int> &prices, int fee) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, cash - static_cast<long long>(p));
        cash = std::max(cash, hold + p - fee);
    }
    return static_cast<int>(cash);
}

int main() {
    std::vector<int> a = {1, 3, 2, 8, 4, 9};
    std::vector<int> b = {1, 3, 7, 5, 10, 3};
    std::vector<int> c = {1};
    std::vector<int> d = {5, 4, 3};
    std::vector<int> e = {1, 4, 6};
    assert(maxProfitFee(a, 2) == 8);
    assert(maxProfitFee(b, 3) == 6);
    assert(maxProfitFee(c, 1) == 0);
    assert(maxProfitFee(d, 1) == 0);
    assert(maxProfitFee(e, 2) == 3);
    std::cout << "best_time_to_buy_and_sell_stock_with_fee: all tests passed\n";
    return 0;
}
