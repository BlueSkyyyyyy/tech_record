// 647. 回文子串
// 见 palindromic_substrings.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int countSubstrings(const std::string &s) {
    int n = static_cast<int>(s.size());
    std::vector<std::vector<bool>> dp(n, std::vector<bool>(n, false));
    int count = 0;
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            if (s[i] == s[j] && (j - i < 2 || dp[i + 1][j - 1])) {
                dp[i][j] = true;
                ++count;
            }
        }
    }
    return count;
}

int main() {
    assert(countSubstrings("abc") == 3);
    assert(countSubstrings("aaa") == 6);
    assert(countSubstrings("a") == 1);
    assert(countSubstrings("") == 0);
    assert(countSubstrings("abba") == 6);
    std::cout << "palindromic_substrings: all tests passed\n";
    return 0;
}
