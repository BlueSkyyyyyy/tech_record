// 205. 同构字符串（双向哈希映射）
// 见 isomorphic_strings.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>

bool isIsomorphic(const std::string &s, const std::string &t) {
    if (s.size() != t.size()) return false;
    std::unordered_map<char, char> forward, backward;
    for (std::size_t i = 0; i < s.size(); ++i) {
        auto fit = forward.find(s[i]);
        if (fit != forward.end() && fit->second != t[i]) return false;
        auto bit = backward.find(t[i]);
        if (bit != backward.end() && bit->second != s[i]) return false;
        forward[s[i]] = t[i];
        backward[t[i]] = s[i];
    }
    return true;
}

int main() {
    assert(isIsomorphic("egg", "add") == true);
    assert(isIsomorphic("foo", "bar") == false);
    assert(isIsomorphic("paper", "title") == true);
    assert(isIsomorphic("ab", "aa") == false);
    assert(isIsomorphic("", "") == true);
    std::cout << "isomorphic_strings: all tests passed\n";
    return 0;
}
