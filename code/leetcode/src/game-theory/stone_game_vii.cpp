// 1690. 石子游戏 VII
// 见 stone_game_vii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int stoneGameVII(std::vector<int> stones) {
    int n = stones.size();
    std::vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + stones[i];
    }
    auto rangeSum = [&](int i, int j) { return prefix[j + 1] - prefix[i]; };

    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            int left = rangeSum(i + 1, j) - dp[i + 1][j];
            int right = rangeSum(i, j - 1) - dp[i][j - 1];
            dp[i][j] = std::max(left, right);
        }
    }
    return dp[0][n - 1];
}

int main() {
    std::vector<int> a1{5, 3, 1, 4, 2};
    std::vector<int> a2{7, 90, 5, 1, 100, 10, 10, 0};
    std::vector<int> a3{1, 1};
    std::vector<int> a4{1, 2, 3};
    assert(stoneGameVII(a1) == 6);
    assert(stoneGameVII(a2) == 122);
    assert(stoneGameVII(a3) == 1);
    assert(stoneGameVII(a4) == 2);

    std::cout << "stone_game_vii: all tests passed\n";
    return 0;
}
