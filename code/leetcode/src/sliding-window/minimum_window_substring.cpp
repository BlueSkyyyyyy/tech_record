// 76. 最小覆盖子串（变长滑动窗口 + 计数）
// 见 minimum_window_substring.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>
#include <string>
#include <unordered_map>

std::string minWindow(const std::string &s, const std::string &t) {
    if (s.empty() || t.empty()) return "";
    std::unordered_map<char, int> need, window;
    for (char c : t) need[c]++;
    int formed = 0;
    int left = 0, bestLen = INT_MAX, bestLeft = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        char c = s[right];
        window[c]++;
        if (need.count(c) && window[c] == need[c]) ++formed;
        while (formed == static_cast<int>(need.size())) {
            if (right - left + 1 < bestLen) {
                bestLen = right - left + 1;
                bestLeft = left;
            }
            char lc = s[left];
            window[lc]--;
            if (need.count(lc) && window[lc] < need[lc]) --formed;
            ++left;
        }
    }
    return bestLen == INT_MAX ? "" : s.substr(bestLeft, bestLen);
}

int main() {
    assert(minWindow("ADOBECODEBANC", "ABC") == "BANC");
    assert(minWindow("a", "a") == "a");
    assert(minWindow("a", "aa") == "");
    assert(minWindow("aa", "aa") == "aa");
    assert(minWindow("cabwefgewcwaefgcf", "cae") == "cwae");
    assert(minWindow("abc", "") == "");
    std::cout << "minimum_window_substring: all tests passed\n";
    return 0;
}
