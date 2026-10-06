// 1143. 最长公共子序列
// 见 longest_common_subsequence.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int longestCommonSubsequence(const std::string &text1, const std::string &text2) {
    int m = static_cast<int>(text1.size());
    int n = static_cast<int>(text2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}

int main() {
    assert(longestCommonSubsequence("abcde", "ace") == 3);
    assert(longestCommonSubsequence("abc", "abc") == 3);
    assert(longestCommonSubsequence("abc", "def") == 0);
    assert(longestCommonSubsequence("", "abc") == 0);
    assert(longestCommonSubsequence("bl", "yby") == 1);
    std::cout << "longest_common_subsequence: all tests passed\n";
    return 0;
}
