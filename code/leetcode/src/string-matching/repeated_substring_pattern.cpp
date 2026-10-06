// 459. 重复的子字符串
// 见 repeated_substring_pattern.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

bool repeatedSubstringPattern(const std::string &s) {
    int n = static_cast<int>(s.size());
    int last = buildLps(s).back();
    int period = n - last;
    return last > 0 && n % period == 0;
}

int main() {
    assert(repeatedSubstringPattern("abab") == true);
    assert(repeatedSubstringPattern("aba") == false);
    assert(repeatedSubstringPattern("abcabcabcabc") == true);
    assert(repeatedSubstringPattern("a") == false);
    assert(repeatedSubstringPattern("aaaa") == true);
    assert(repeatedSubstringPattern("abcd") == false);
    assert(repeatedSubstringPattern("ababab") == true);

    std::cout << "repeatedSubstringPattern: all tests passed\n";
    return 0;
}
