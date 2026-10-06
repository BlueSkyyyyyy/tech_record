// 686. 重复叠加字符串匹配
// 见 repeated_string_match.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>

int repeatedStringMatch(const std::string &a, const std::string &b) {
    int times = static_cast<int>((b.size() + a.size() - 1) / a.size());
    for (int t = times; t <= times + 1; ++t) {
        std::string rep;
        for (int i = 0; i < t; ++i) {
            rep += a;
        }
        if (rep.find(b) != std::string::npos) {
            return t;
        }
    }
    return -1;
}

int main() {
    assert(repeatedStringMatch("abcd", "cdabcdab") == 3);
    assert(repeatedStringMatch("a", "aa") == 2);
    assert(repeatedStringMatch("abc", "cabcabca") == 4);
    assert(repeatedStringMatch("abc", "wxyz") == -1);
    assert(repeatedStringMatch("aa", "a") == 1);
    assert(repeatedStringMatch("ab", "ba") == 2);

    std::cout << "repeatedStringMatch: all tests passed\n";
    return 0;
}
