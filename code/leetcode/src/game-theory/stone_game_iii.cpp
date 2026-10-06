// 1406. 石子游戏 III
// 见 stone_game_iii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

bool stoneGameIII(std::vector<int> stoneValue) {
    int n = stoneValue.size();
    std::vector<int> suffix(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        suffix[i] = suffix[i + 1] + stoneValue[i];
    }
    std::vector<int> dp(n + 4, 0);
    for (int i = n - 1; i >= 0; --i) {
        int best = INT_MIN;
        for (int x = 1; x <= 3; ++x) {
            if (i + x <= n) {
                int take = suffix[i] - suffix[i + x];
                best = std::max(best, take - dp[i + x]);
            }
        }
        dp[i] = best;
    }
    return dp[0] > 0;
}

int main() {
    std::vector<int> a1{1, 2, 3, 7};
    std::vector<int> a2{1, 2, 3, -9};
    std::vector<int> a3{1, 2, 3, 6};
    std::vector<int> a4{1, 2, 3};
    std::vector<int> a5{1};
    assert(stoneGameIII(a1) == false);
    assert(stoneGameIII(a2) == true);
    assert(stoneGameIII(a3) == false);
    assert(stoneGameIII(a4) == true);
    assert(stoneGameIII(a5) == true);

    std::cout << "stone_game_iii: all tests passed\n";
    return 0;
}
