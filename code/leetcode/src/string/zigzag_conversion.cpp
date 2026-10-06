// 6. Z 字形变换
// 见 zigzag_conversion.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string convert(std::string s, int numRows) {
    int n = static_cast<int>(s.size());
    if (numRows == 1 || numRows >= n) return s;

    std::vector<std::string> rows(numRows);
    int cur = 0;
    int step = 1;
    for (char ch : s) {
        rows[cur] += ch;
        if (cur == 0) {
            step = 1;
        } else if (cur == numRows - 1) {
            step = -1;
        }
        cur += step;
    }

    std::string result;
    for (const std::string &row : rows) result += row;
    return result;
}

int main() {
    assert(convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR");
    assert(convert("PAYPALISHIRING", 4) == "PINALSIGYAHRPI");
    assert(convert("A", 1) == "A");
    assert(convert("AB", 1) == "AB");
    assert(convert("ABC", 5) == "ABC");
    assert(convert("HELLO", 2) == "HLOEL");
    std::cout << "zigzag_conversion: all tests passed\n";
    return 0;
}
