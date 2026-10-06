// 395. 至少有 K 个重复字符的最长子串
// 见 longest_substring_with_at_least_k_repeating.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

int longestSubstring(const std::string &s, int k) {
    if (static_cast<int>(s.size()) < k) return 0;

    int cnt[26] = {0};
    for (char c : s) ++cnt[c - 'a'];

    char bad = 0;
    for (int i = 0; i < 26; ++i) {
        if (cnt[i] > 0 && cnt[i] < k) {
            bad = static_cast<char>('a' + i);
            break;
        }
    }
    if (bad == 0) return static_cast<int>(s.size());

    int best = 0, start = 0;
    for (int i = 0; i <= static_cast<int>(s.size()); ++i) {
        if (i == static_cast<int>(s.size()) || s[i] == bad) {
            if (i > start) best = std::max(best, longestSubstring(s.substr(start, i - start), k));
            start = i + 1;
        }
    }
    return best;
}

int main() {
    assert(longestSubstring("aaabb", 3) == 3);
    assert(longestSubstring("ababbc", 2) == 5);
    assert(longestSubstring("aaabbb", 3) == 6);
    assert(longestSubstring("abc", 2) == 0);
    assert(longestSubstring("a", 1) == 1);
    assert(longestSubstring("ababacb", 3) == 0);
    std::cout << "longest_substring: all tests passed\n";
    return 0;
}
