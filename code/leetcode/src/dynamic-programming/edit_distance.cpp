// 72. 编辑距离
// 见 edit_distance.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int minDistance(const std::string &word1, const std::string &word2) {
    int m = static_cast<int>(word1.size());
    int n = static_cast<int>(word2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    return dp[m][n];
}

int main() {
    assert(minDistance("horse", "ros") == 3);
    assert(minDistance("intention", "execution") == 5);
    assert(minDistance("", "") == 0);
    assert(minDistance("abc", "") == 3);
    assert(minDistance("", "abc") == 3);
    assert(minDistance("abc", "abc") == 0);
    std::cout << "edit_distance: all tests passed\n";
    return 0;
}
