// 28. 找出字符串中第一个匹配项的下标 · KMP
// 见 implement_strstr.py 的题目与思路说明。
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

int strStr(const std::string &haystack, const std::string &needle) {
    if (needle.empty()) {
        return 0;
    }
    std::vector<int> lps = buildLps(needle);
    int m = static_cast<int>(needle.size()), j = 0;
    for (int i = 0; i < static_cast<int>(haystack.size()); ++i) {
        while (j > 0 && haystack[i] != needle[j]) {
            j = lps[j - 1];
        }
        if (haystack[i] == needle[j]) {
            ++j;
        }
        if (j == m) {
            return i - m + 1;
        }
    }
    return -1;
}

int main() {
    std::vector<int> wantLps = {0, 0, 1, 2, 3, 0, 1};
    assert(buildLps("ababaca") == wantLps);
    assert(strStr("sadbutsad", "sad") == 0);
    assert(strStr("leetcode", "leeto") == -1);
    assert(strStr("hello", "") == 0);
    assert(strStr("aabaabaaa", "aabaaa") == 3);
    assert(strStr("mississippi", "issip") == 4);
    assert(strStr("abc", "abcd") == -1);

    std::cout << "strStr: all tests passed\n";
    return 0;
}
