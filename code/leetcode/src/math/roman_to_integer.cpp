// 13. 罗马数字转整数
// 见 roman_to_integer.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>

int romanToInteger(const std::string &s) {
    std::unordered_map<char, int> values = {
        {'I', 1},   {'V', 5},   {'X', 10},  {'L', 50},
        {'C', 100}, {'D', 500}, {'M', 1000}};
    int total = 0;
    int n = static_cast<int>(s.size());
    for (int i = 0; i < n; ++i) {
        int v = values[s[i]];
        if (i + 1 < n && v < values[s[i + 1]]) {
            total -= v;
        } else {
            total += v;
        }
    }
    return total;
}

int main() {
    assert(romanToInteger("III") == 3);
    assert(romanToInteger("IV") == 4);
    assert(romanToInteger("IX") == 9);
    assert(romanToInteger("LVIII") == 58);
    assert(romanToInteger("MCMXCIV") == 1994);
    assert(romanToInteger("XL") == 40);
    assert(romanToInteger("CDXLIV") == 444);
    assert(romanToInteger("MMMCMXCIX") == 3999);

    std::cout << "roman_to_integer: all tests passed\n";
    return 0;
}
