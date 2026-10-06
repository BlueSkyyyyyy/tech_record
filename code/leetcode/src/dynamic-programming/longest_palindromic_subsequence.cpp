// 516. 最长回文子序列
// 见 longest_palindromic_subsequence.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int longestPalindromeSubseq(const std::string &s) {
    int n = static_cast<int>(s.size());
    if (n == 0) return 0;
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = n - 1; i >= 0; --i) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = std::max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[0][n - 1];
}

int main() {
    assert(longestPalindromeSubseq("bbbab") == 4);
    assert(longestPalindromeSubseq("cbbd") == 2);
    assert(longestPalindromeSubseq("a") == 1);
    assert(longestPalindromeSubseq("") == 0);
    assert(longestPalindromeSubseq("abcde") == 1);
    std::cout << "longest_palindromic_subsequence: all tests passed\n";
    return 0;
}
