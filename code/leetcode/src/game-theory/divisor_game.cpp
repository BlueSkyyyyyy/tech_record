// 1025. 除数博弈
// 见 divisor_game.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool divisorGame(int n) {
    std::vector<bool> dp(n + 1, false);
    for (int i = 2; i <= n; ++i) {
        for (int x = 1; x < i; ++x) {
            if (i % x == 0 && !dp[i - x]) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}

int main() {
    assert(divisorGame(1) == false);
    assert(divisorGame(2) == true);
    assert(divisorGame(3) == false);
    assert(divisorGame(4) == true);
    assert(divisorGame(5) == false);
    assert(divisorGame(1000) == true);
    for (int n = 1; n < 60; ++n) {
        assert(divisorGame(n) == (n % 2 == 0));
    }

    std::cout << "divisor_game: all tests passed\n";
    return 0;
}
