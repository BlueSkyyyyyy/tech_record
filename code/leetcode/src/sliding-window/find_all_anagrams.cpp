// 438. 找到字符串中所有字母异位词（定长滑动窗口 + 计数）
// 见 find_all_anagrams.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> findAnagrams(const std::string &s, const std::string &p) {
    std::vector<int> res;
    int n = static_cast<int>(s.size()), m = static_cast<int>(p.size());
    if (n < m) return res;
    std::array<int, 26> need{}, window{};
    for (char c : p) need[c - 'a']++;
    for (int i = 0; i < n; ++i) {
        window[s[i] - 'a']++;
        if (i >= m) window[s[i - m] - 'a']--;
        if (i >= m - 1 && window == need) res.push_back(i - m + 1);
    }
    return res;
}

int main() {
    const std::vector<int> want1 = {0, 6};
    const std::vector<int> want2 = {0, 1, 2};
    const std::vector<int> want3 = {1};
    const std::vector<int> want4;
    const std::vector<int> want5 = {0, 1, 2};
    assert(findAnagrams("cbaebabacd", "abc") == want1);
    assert(findAnagrams("abab", "ab") == want2);
    assert(findAnagrams("a", "ab") == want4);
    assert(findAnagrams("baa", "aa") == want3);
    assert(findAnagrams("aaaa", "aa") == want5);
    std::cout << "find_all_anagrams: all tests passed\n";
    return 0;
}
