// 541. 反转字符串 II
// 见 reverse_string_ii.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

std::string reverseStr(std::string s, int k) {
    int n = static_cast<int>(s.size());
    for (int i = 0; i < n; i += 2 * k) {
        int left = i;
        int right = std::min(i + k, n) - 1;
        while (left < right) {
            std::swap(s[left], s[right]);
            ++left;
            --right;
        }
    }
    return s;
}

int main() {
    assert(reverseStr("abcdefg", 2) == "bacdfeg");
    assert(reverseStr("abcd", 2) == "bacd");
    assert(reverseStr("abcdefg", 8) == "gfedcba");
    assert(reverseStr("a", 2) == "a");
    assert(reverseStr("", 3) == "");
    assert(reverseStr("abcd", 4) == "dcba");
    assert(reverseStr("abcdef", 3) == "cbadef");
    assert(reverseStr("ab", 1) == "ab");
    std::cout << "reverse_string_ii: all tests passed\n";
    return 0;
}
