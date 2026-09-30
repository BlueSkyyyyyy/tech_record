// 125. 验证回文串（对撞双指针）
// 见 valid_palindrome.py 的题目与思路说明。
#include <cassert>
#include <cctype>
#include <iostream>
#include <string>

bool isPalindrome(const std::string &s) {
    int lo = 0, hi = static_cast<int>(s.size()) - 1;
    while (lo < hi) {
        while (lo < hi && !std::isalnum(static_cast<unsigned char>(s[lo]))) ++lo;
        while (lo < hi && !std::isalnum(static_cast<unsigned char>(s[hi]))) --hi;
        if (std::tolower(static_cast<unsigned char>(s[lo])) !=
            std::tolower(static_cast<unsigned char>(s[hi]))) {
            return false;
        }
        ++lo;
        --hi;
    }
    return true;
}

int main() {
    assert(isPalindrome("A man, a plan, a canal: Panama"));
    assert(!isPalindrome("race a car"));
    assert(isPalindrome(" "));
    assert(isPalindrome(""));
    assert(!isPalindrome("0P"));
    std::cout << "valid_palindrome: all tests passed\n";
    return 0;
}
