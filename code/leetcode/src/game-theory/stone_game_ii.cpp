// 1140. 石子游戏 II
// 见 stone_game_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <vector>

int stoneGameII(std::vector<int> piles) {
    int n = piles.size();
    std::vector<int> suffix(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        suffix[i] = suffix[i + 1] + piles[i];
    }
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, -1));
    std::function<int(int, int)> f = [&](int i, int m) -> int {
        if (i >= n) {
            return 0;
        }
        if (dp[i][m] != -1) {
            return dp[i][m];
        }
        int best = 0;
        for (int x = 1; x <= 2 * m; ++x) {
            if (i + x > n) {
                break;
            }
            best = std::max(best, suffix[i] - f(i + x, std::max(m, x)));
        }
        return dp[i][m] = best;
    };
    return f(0, 1);
}

int main() {
    std::vector<int> a1{2, 7, 9, 4, 4};
    std::vector<int> a2{1, 2, 3};
    std::vector<int> a3{1};
    std::vector<int> a4{1, 2, 3, 4, 5, 6};
    assert(stoneGameII(a1) == 10);
    assert(stoneGameII(a2) == 3);
    assert(stoneGameII(a3) == 1);
    assert(stoneGameII(a4) == 10);

    std::cout << "stone_game_ii: all tests passed\n";
    return 0;
}
