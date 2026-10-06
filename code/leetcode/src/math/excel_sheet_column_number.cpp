// 171. Excel 表列序号
// 见 excel_sheet_column_number.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>

int excelSheetColumnNumber(const std::string &s) {
    int res = 0;
    for (char ch : s) {
        res = res * 26 + (ch - 'A' + 1);
    }
    return res;
}

int main() {
    assert(excelSheetColumnNumber("A") == 1);
    assert(excelSheetColumnNumber("B") == 2);
    assert(excelSheetColumnNumber("Z") == 26);
    assert(excelSheetColumnNumber("AA") == 27);
    assert(excelSheetColumnNumber("AB") == 28);
    assert(excelSheetColumnNumber("ZY") == 701);
    assert(excelSheetColumnNumber("AZ") == 52);
    assert(excelSheetColumnNumber("ZZZ") == 18278);

    std::cout << "excel_sheet_column_number: all tests passed\n";
    return 0;
}
