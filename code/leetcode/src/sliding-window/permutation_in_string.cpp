// 567. 字符串的排列（定长滑动窗口 + 计数）
// 见 permutation_in_string.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>

bool checkInclusion(const std::string &s1, const std::string &s2) {
    int n1 = static_cast<int>(s1.size()), n2 = static_cast<int>(s2.size());
    if (n1 > n2) return false;
    std::array<int, 26> need{}, window{};
    for (char c : s1) need[c - 'a']++;
    for (int i = 0; i < n2; ++i) {
        window[s2[i] - 'a']++;
        if (i >= n1) window[s2[i - n1] - 'a']--;
        if (i >= n1 - 1 && window == need) return true;
    }
    return false;
}

int main() {
    assert(checkInclusion("ab", "eidbaooo") == true);
    assert(checkInclusion("ab", "eidboaoo") == false);
    assert(checkInclusion("abc", "bbbca") == true);
    assert(checkInclusion("hello", "ooolleoooleh") == false);
    assert(checkInclusion("a", "a") == true);
    std::cout << "permutation_in_string: all tests passed\n";
    return 0;
}
