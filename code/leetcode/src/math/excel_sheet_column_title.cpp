// 168. Excel 表列名称
// 见 excel_sheet_column_title.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>

std::string excelSheetColumnTitle(int columnNumber) {
    std::string res;
    int n = columnNumber;
    while (n > 0) {
        n -= 1;
        res.push_back(static_cast<char>('A' + n % 26));
        n /= 26;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

int main() {
    assert(excelSheetColumnTitle(1) == "A");
    assert(excelSheetColumnTitle(2) == "B");
    assert(excelSheetColumnTitle(26) == "Z");
    assert(excelSheetColumnTitle(27) == "AA");
    assert(excelSheetColumnTitle(28) == "AB");
    assert(excelSheetColumnTitle(701) == "ZY");
    assert(excelSheetColumnTitle(52) == "AZ");
    assert(excelSheetColumnTitle(18278) == "ZZZ");

    std::cout << "excel_sheet_column_title: all tests passed\n";
    return 0;
}
