// 121. 买卖股票的最佳时机
// 见 best_time_to_buy_and_sell_stock.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int maxProfit(const std::vector<int> &prices) {
    int minPrice = INT_MAX;
    int best = 0;
    for (int p : prices) {
        best = std::max(best, p - minPrice);
        minPrice = std::min(minPrice, p);
    }
    return best;
}

int main() {
    std::vector<int> a = {7, 1, 5, 3, 6, 4};
    assert(maxProfit(a) == 5);
    std::vector<int> b = {7, 6, 4, 3, 1};
    assert(maxProfit(b) == 0);
    std::vector<int> c = {1};
    assert(maxProfit(c) == 0);
    std::vector<int> d = {2, 4, 1};
    assert(maxProfit(d) == 2);
    std::vector<int> e = {3, 3, 5, 0, 0, 3, 1, 4};
    assert(maxProfit(e) == 4);
    std::cout << "best_time_to_buy_and_sell_stock: all tests passed\n";
    return 0;
}
