// 28. 找出字符串中第一个匹配项的下标（KMP）
// 见 implement_strstr.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

int strStr(const std::string &haystack, const std::string &needle) {
    if (needle.empty()) return 0;
    int n = static_cast<int>(haystack.size());
    int m = static_cast<int>(needle.size());

    std::vector<int> lps(m, 0);
    int k = 0;
    for (int i = 1; i < m; ++i) {
        while (k > 0 && needle[i] != needle[k]) k = lps[k - 1];
        if (needle[i] == needle[k]) ++k;
        lps[i] = k;
    }

    k = 0;
    for (int i = 0; i < n; ++i) {
        while (k > 0 && haystack[i] != needle[k]) k = lps[k - 1];
        if (haystack[i] == needle[k]) {
            ++k;
            if (k == m) return i - m + 1;
        }
    }
    return -1;
}

int main() {
    assert(strStr("sadbutsad", "sad") == 0);
    assert(strStr("leetcode", "leeto") == -1);
    assert(strStr("hello", "") == 0);
    assert(strStr("a", "a") == 0);
    assert(strStr("abcabcabd", "abcabd") == 3);
    assert(strStr("aaaaa", "bba") == -1);
    assert(strStr("mississippi", "issip") == 4);

    std::cout << "implement_strstr: all tests passed\n";
    return 0;
}
