// 5. 最长回文子串 · Manacher
// 见 longest_palindromic_substring.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string longestPalindrome(const std::string &s) {
    if (s.size() <= 1) {
        return s;
    }
    std::string t = "^#";
    for (char c : s) {
        t += c;
        t += '#';
    }
    t += "$";
    int n = static_cast<int>(t.size());
    std::vector<int> p(n, 0);
    int center = 0, right = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (i < right) {
            p[i] = std::min(right - i, p[2 * center - i]);
        }
        while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
            ++p[i];
        }
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }
    int maxLen = 0, centerIdx = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] > maxLen) {
            maxLen = p[i];
            centerIdx = i;
        }
    }
    int start = (centerIdx - maxLen) / 2;
    return s.substr(start, maxLen);
}

int main() {
    std::string r = longestPalindrome("babad");
    assert(r == "bab" || r == "aba");
    assert(longestPalindrome("cbbd") == "bb");
    assert(longestPalindrome("a") == "a");
    assert(longestPalindrome("") == "");
    assert(longestPalindrome("abb") == "bb");
    assert(longestPalindrome("abccba") == "abccba");
    assert(longestPalindrome("forgeeksskeegfor") == "geeksskeeg");

    std::cout << "longestPalindrome: all tests passed\n";
    return 0;
}
