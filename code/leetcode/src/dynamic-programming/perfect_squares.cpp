// 279. 完全平方数
// 见 perfect_squares.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int numSquares(int n) {
    std::vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        dp[i] = i;
        for (int j = 1; j * j <= i; ++j) {
            dp[i] = std::min(dp[i], dp[i - j * j] + 1);
        }
    }
    return dp[n];
}

int main() {
    assert(numSquares(12) == 3);
    assert(numSquares(13) == 2);
    assert(numSquares(1) == 1);
    assert(numSquares(0) == 0);
    assert(numSquares(7) == 4);
    std::cout << "perfect_squares: all tests passed\n";
    return 0;
}
