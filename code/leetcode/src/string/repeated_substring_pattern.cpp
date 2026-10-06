// 459. 重复的子字符串
// 见 repeated_substring_pattern.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

bool repeatedSubstringPattern(const std::string &s) {
    int n = static_cast<int>(s.size());
    if (n == 0) return false;

    std::vector<int> lps(n, 0);
    int k = 0;
    for (int i = 1; i < n; ++i) {
        while (k > 0 && s[i] != s[k]) k = lps[k - 1];
        if (s[i] == s[k]) ++k;
        lps[i] = k;
    }

    int p = lps[n - 1];
    return p > 0 && n % (n - p) == 0;
}

int main() {
    assert(repeatedSubstringPattern("abab") == true);
    assert(repeatedSubstringPattern("aba") == false);
    assert(repeatedSubstringPattern("abcabcabcabc") == true);
    assert(repeatedSubstringPattern("a") == false);
    assert(repeatedSubstringPattern("aa") == true);
    assert(repeatedSubstringPattern("abc") == false);
    assert(repeatedSubstringPattern("abcab") == false);

    std::cout << "repeated_substring_pattern: all tests passed\n";
    return 0;
}
