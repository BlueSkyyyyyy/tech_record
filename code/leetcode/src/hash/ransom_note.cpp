// 383. 赎金信（字符计数）
// 见 ransom_note.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>

bool canConstruct(const std::string &ransomNote, const std::string &magazine) {
    std::unordered_map<char, int> count;
    for (char ch : magazine) ++count[ch];
    for (char ch : ransomNote) {
        if (count[ch] == 0) return false;
        --count[ch];
    }
    return true;
}

int main() {
    assert(canConstruct("a", "b") == false);
    assert(canConstruct("aa", "ab") == false);
    assert(canConstruct("aa", "aab") == true);
    assert(canConstruct("", "anything") == true);
    assert(canConstruct("abc", "cba") == true);
    std::cout << "ransom_note: all tests passed\n";
    return 0;
}
