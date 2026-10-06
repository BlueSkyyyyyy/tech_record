// 8. 字符串转换整数 atoi
// 见 string_to_integer_atoi.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>
#include <string>

int myAtoi(const std::string &s) {
    int n = static_cast<int>(s.size());
    int i = 0;
    while (i < n && s[i] == ' ') ++i;

    int sign = 1;
    if (i < n && (s[i] == '+' || s[i] == '-')) {
        sign = (s[i] == '-') ? -1 : 1;
        ++i;
    }

    long long num = 0;
    while (i < n && s[i] >= '0' && s[i] <= '9') {
        num = num * 10 + (s[i] - '0');
        if (sign * num > INT_MAX) return INT_MAX;
        if (sign * num < INT_MIN) return INT_MIN;
        ++i;
    }

    return static_cast<int>(sign * num);
}

int main() {
    assert(myAtoi("42") == 42);
    assert(myAtoi("   -42") == -42);
    assert(myAtoi("4193 with words") == 4193);
    assert(myAtoi("words and 987") == 0);
    assert(myAtoi("-91283472332") == INT_MIN);
    assert(myAtoi("2147483648") == INT_MAX);
    assert(myAtoi("+1") == 1);
    assert(myAtoi("   +0 123") == 0);
    assert(myAtoi("0000123") == 123);
    assert(myAtoi("") == 0);
    assert(myAtoi("+-12") == 0);
    std::cout << "string_to_integer_atoi: all tests passed\n";
    return 0;
}
