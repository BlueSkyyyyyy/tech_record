// 647. 回文子串 · Manacher
// 见 palindromic_substrings.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int countSubstrings(const std::string &s) {
    if (s.empty()) {
        return 0;
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
    int total = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (i % 2 == 0) {
            total += p[i] / 2 + 1;  // 原串字符为中心：奇数长度
        } else {
            total += (p[i] + 1) / 2;  // '#' 为中心：偶数长度
        }
    }
    return total;
}

int main() {
    assert(countSubstrings("abc") == 3);
    assert(countSubstrings("aaa") == 6);
    assert(countSubstrings("aba") == 4);
    assert(countSubstrings("a") == 1);
    assert(countSubstrings("") == 0);
    assert(countSubstrings("aaaa") == 10);

    std::cout << "countSubstrings: all tests passed\n";
    return 0;
}
