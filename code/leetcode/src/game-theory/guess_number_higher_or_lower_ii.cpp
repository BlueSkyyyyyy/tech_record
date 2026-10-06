// 375. 猜数字大小 II
// 见 guess_number_higher_or_lower_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int getMoneyAmount(int n) {
    std::vector<std::vector<int>> dp(n + 2, std::vector<int>(n + 2, 0));
    for (int len = 2; len <= n; ++len) {
        for (int i = 1; i + len - 1 <= n; ++i) {
            int j = i + len - 1;
            int best = INT_MAX;
            for (int x = i; x <= j; ++x) {
                int cost = x + std::max(dp[i][x - 1], dp[x + 1][j]);
                best = std::min(best, cost);
            }
            dp[i][j] = best;
        }
    }
    return dp[1][n];
}

int main() {
    assert(getMoneyAmount(1) == 0);
    assert(getMoneyAmount(2) == 1);
    assert(getMoneyAmount(3) == 2);
    assert(getMoneyAmount(4) == 4);
    assert(getMoneyAmount(10) == 16);

    std::cout << "guess_number_higher_or_lower_ii: all tests passed\n";
    return 0;
}
