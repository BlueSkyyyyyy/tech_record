// 1392. 最长快乐前缀
// 见 longest_happy_prefix.py 的题目与思路说明。
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

std::string longestPrefix(const std::string &s) {
    if (s.empty()) {
        return "";
    }
    int k = buildLps(s).back();
    return s.substr(0, k);
}

int main() {
    assert(longestPrefix("level") == "l");
    assert(longestPrefix("ababab") == "abab");
    assert(longestPrefix("leetcodeleet") == "leet");
    assert(longestPrefix("a") == "");
    assert(longestPrefix("abcd") == "");
    assert(longestPrefix("aaaa") == "aaa");

    std::cout << "longestPrefix: all tests passed\n";
    return 0;
}
