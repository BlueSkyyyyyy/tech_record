// 309. 买卖股票的最佳时机含冷冻期
// 见 best_time_to_buy_and_sell_stock_with_cooldown.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxProfitCooldown(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, sold = NEG, rest = 0;
    for (int p : prices) {
        long long new_hold = std::max(hold, rest - p);
        long long new_sold = hold + p;
        long long new_rest = std::max(rest, sold);
        hold = new_hold;
        sold = new_sold;
        rest = new_rest;
    }
    return static_cast<int>(std::max(rest, sold));
}

int main() {
    std::vector<int> a = {1, 2, 3, 0, 2};
    std::vector<int> b = {1};
    std::vector<int> c = {2, 1, 4};
    std::vector<int> d = {6, 1, 3, 2, 4, 7};
    assert(maxProfitCooldown(a) == 3);
    assert(maxProfitCooldown(b) == 0);
    assert(maxProfitCooldown(c) == 3);
    assert(maxProfitCooldown(d) == 6);
    std::cout << "best_time_to_buy_and_sell_stock_with_cooldown: all tests passed\n";
    return 0;
}
