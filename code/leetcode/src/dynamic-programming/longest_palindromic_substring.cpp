// 5. 最长回文子串
// 见 longest_palindromic_substring.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

std::string longestPalindrome(const std::string &s) {
    if (s.empty()) return "";
    int n = static_cast<int>(s.size());

    auto expand = [&](int left, int right) {
        while (left >= 0 && right < n && s[left] == s[right]) {
            --left;
            ++right;
        }
        return std::make_pair(left + 1, right - 1);
    };

    int start = 0, end = 0;
    for (int i = 0; i < n; ++i) {
        auto odd = expand(i, i);
        auto even = expand(i, i + 1);
        if (odd.second - odd.first > end - start) {
            start = odd.first;
            end = odd.second;
        }
        if (even.second - even.first > end - start) {
            start = even.first;
            end = even.second;
        }
    }
    return s.substr(start, end - start + 1);
}

int main() {
    std::string a = longestPalindrome("babad");
    assert(a == "bab" || a == "aba");
    assert(longestPalindrome("cbbd") == "bb");
    assert(longestPalindrome("a") == "a");
    std::string c = longestPalindrome("ac");
    assert(c == "a" || c == "c");
    assert(longestPalindrome("") == "");
    std::cout << "longest_palindromic_substring: all tests passed\n";
    return 0;
}
