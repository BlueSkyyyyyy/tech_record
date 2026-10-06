// 214. 最短回文串
// 见 shortest_palindrome.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

std::string shortestPalindrome(const std::string &s) {
    if (s.size() <= 1) {
        return s;
    }
    std::string rev(s.rbegin(), s.rend());
    int k = buildLps(s + "#" + rev).back();
    std::string tail = s.substr(k);
    std::reverse(tail.begin(), tail.end());
    return tail + s;
}

int main() {
    assert(shortestPalindrome("aacecaaa") == "aaacecaaa");
    assert(shortestPalindrome("abcd") == "dcbabcd");
    assert(shortestPalindrome("a") == "a");
    assert(shortestPalindrome("") == "");
    assert(shortestPalindrome("aba") == "aba");

    std::cout << "shortestPalindrome: all tests passed\n";
    return 0;
}
