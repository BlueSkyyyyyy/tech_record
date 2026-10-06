// 877. 石子游戏
// 见 stone_game.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

bool stoneGame(std::vector<int> piles) {
    int n = piles.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        dp[i][i] = piles[i];
    }
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            dp[i][j] = std::max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1] >= 0;
}

int main() {
    std::vector<int> a1{5, 3, 4, 5};
    std::vector<int> a2{3, 7, 2, 3};
    std::vector<int> a3{7, 7, 7, 7};
    std::vector<int> a4{1, 2};
    assert(stoneGame(a1) == true);
    assert(stoneGame(a2) == true);
    assert(stoneGame(a3) == true);
    assert(stoneGame(a4) == true);

    std::cout << "stone_game: all tests passed\n";
    return 0;
}
