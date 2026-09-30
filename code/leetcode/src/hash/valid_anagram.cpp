// 242. 有效的字母异位词（字符计数）
// 见 valid_anagram.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>

bool isAnagram(const std::string &s, const std::string &t) {
    if (s.size() != t.size()) return false;
    std::unordered_map<char, int> count;
    for (char ch : s) ++count[ch];
    for (char ch : t) {
        if (count[ch] == 0) return false;
        --count[ch];
    }
    return true;
}

int main() {
    assert(isAnagram("anagram", "nagaram") == true);
    assert(isAnagram("rat", "car") == false);
    assert(isAnagram("", "") == true);
    assert(isAnagram("aacc", "ccac") == false);
    std::cout << "valid_anagram: all tests passed\n";
    return 0;
}
