// 12. 整数转罗马数字
// 见 integer_to_roman.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

std::string integerToRoman(int num) {
    const std::vector<std::pair<int, std::string>> pairs = {
        {1000, "M"},  {900, "CM"}, {500, "D"}, {400, "CD"}, {100, "C"},
        {90, "XC"},   {50, "L"},   {40, "XL"}, {10, "X"},   {9, "IX"},
        {5, "V"},     {4, "IV"},   {1, "I"}};
    std::string res;
    for (const auto &p : pairs) {
        while (num >= p.first) {
            res += p.second;
            num -= p.first;
        }
    }
    return res;
}

int main() {
    assert(integerToRoman(3) == "III");
    assert(integerToRoman(4) == "IV");
    assert(integerToRoman(9) == "IX");
    assert(integerToRoman(58) == "LVIII");
    assert(integerToRoman(1994) == "MCMXCIV");
    assert(integerToRoman(40) == "XL");
    assert(integerToRoman(444) == "CDXLIV");
    assert(integerToRoman(3999) == "MMMCMXCIX");

    std::cout << "integer_to_roman: all tests passed\n";
    return 0;
}
