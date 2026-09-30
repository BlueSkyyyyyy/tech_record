// 424. 替换后的最长重复字符（变长滑动窗口 + 频次）
// 见 longest_repeating_character_replacement.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>

int characterReplacement(const std::string &s, int k) {
    std::array<int, 26> count{};
    int left = 0, best = 0, maxFreq = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        int idx = s[right] - 'A';
        ++count[idx];
        if (count[idx] > maxFreq) maxFreq = count[idx];
        if ((right - left + 1) - maxFreq > k) {
            --count[s[left] - 'A'];
            ++left;
        }
        if (right - left + 1 > best) best = right - left + 1;
    }
    return best;
}

int main() {
    assert(characterReplacement("AABABBA", 1) == 4);
    assert(characterReplacement("ABAB", 2) == 4);
    assert(characterReplacement("ABAB", 0) == 1);
    assert(characterReplacement("A", 0) == 1);
    assert(characterReplacement("AAAA", 2) == 4);
    assert(characterReplacement("BAAAB", 2) == 5);
    std::cout << "longest_repeating_character_replacement: all tests passed\n";
    return 0;
}
