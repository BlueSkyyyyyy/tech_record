// 121. 买卖股票的最佳时机
// 见 best_time_to_buy_and_sell_stock.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfit(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, -static_cast<long long>(p));
        cash = std::max(cash, hold + p);
    }
    return static_cast<int>(cash);
}

int main() {
    std::vector<int> a = {7, 1, 5, 3, 6, 4};
    std::vector<int> b = {7, 6, 4, 3, 1};
    std::vector<int> c = {1};
    std::vector<int> d = {2, 4, 1};
    assert(maxProfit(a) == 5);
    assert(maxProfit(b) == 0);
    assert(maxProfit(c) == 0);
    assert(maxProfit(d) == 2);
    std::cout << "best_time_to_buy_and_sell_stock: all tests passed\n";
    return 0;
}
