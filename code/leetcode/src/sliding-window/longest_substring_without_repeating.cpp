// 3. 无重复字符的最长子串（滑动窗口 + 记录上次出现位置）
// 见 longest_substring_without_repeating.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>

int lengthOfLongestSubstring(const std::string &s) {
    std::unordered_map<char, int> last;
    int left = 0, best = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        char ch = s[right];
        auto it = last.find(ch);
        if (it != last.end() && it->second >= left) left = it->second + 1;
        last[ch] = right;
        best = std::max(best, right - left + 1);
    }
    return best;
}

int main() {
    assert(lengthOfLongestSubstring("abcabcbb") == 3);
    assert(lengthOfLongestSubstring("bbbbb") == 1);
    assert(lengthOfLongestSubstring("pwwkew") == 3);
    assert(lengthOfLongestSubstring("") == 0);
    assert(lengthOfLongestSubstring("dvdf") == 3);
    assert(lengthOfLongestSubstring("abba") == 2);
    std::cout << "longest_substring_without_repeating: all tests passed\n";
    return 0;
}
