// 174. 地下城游戏（倒推 DP + 一维滚动数组空间压缩）
// 见 dungeon_game.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

int calculateMinimumHP(const std::vector<std::vector<int>> &dungeon) {
    int m = static_cast<int>(dungeon.size());
    int n = static_cast<int>(dungeon[0].size());
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    for (int i = m - 1; i >= 0; --i) {
        for (int j = n - 1; j >= 0; --j) {
            int need;
            if (i == m - 1 && j == n - 1)
                need = 1 - dungeon[i][j];
            else if (i == m - 1)
                need = dp[i][j + 1] - dungeon[i][j];
            else if (j == n - 1)
                need = dp[i + 1][j] - dungeon[i][j];
            else
                need = std::min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];
            dp[i][j] = std::max(1, need);
        }
    }
    return dp[0][0];
}

int calculateMinimumHP1D(const std::vector<std::vector<int>> &dungeon) {
    int m = static_cast<int>(dungeon.size());
    int n = static_cast<int>(dungeon[0].size());
    const int INF = std::numeric_limits<int>::max();
    std::vector<int> dp(n + 1, INF);
    dp[n - 1] = 1;
    for (int i = m - 1; i >= 0; --i) {
        for (int j = n - 1; j >= 0; --j) {
            dp[j] = std::max(1, std::min(dp[j], dp[j + 1]) - dungeon[i][j]);
        }
        dp[n] = INF;
    }
    return dp[0];
}

int main() {
    std::vector<std::vector<int>> d{{-2, -3, 3}, {-5, -10, 1}, {10, 30, -5}};
    assert(calculateMinimumHP(d) == 7);
    assert(calculateMinimumHP1D(d) == 7);
    assert(calculateMinimumHP({{0}}) == 1);
    assert(calculateMinimumHP({{-3, 5}}) == 4);
    assert(calculateMinimumHP1D({{100}}) == 1);
    std::cout << "dungeon_game: all tests passed\n";
    return 0;
}
