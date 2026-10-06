// 43. 字符串相乘
// 见 multiply_strings.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

std::string multiply(const std::string &num1, const std::string &num2) {
    if (num1 == "0" || num2 == "0") return "0";

    int m = static_cast<int>(num1.size());
    int n = static_cast<int>(num2.size());
    std::vector<int> res(m + n, 0);

    for (int i = m - 1; i >= 0; --i) {
        for (int j = n - 1; j >= 0; --j) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int p1 = i + j;
            int p2 = i + j + 1;
            int total = mul + res[p2];
            res[p2] = total % 10;
            res[p1] += total / 10;
        }
    }

    int start = 0;
    while (start < static_cast<int>(res.size()) && res[start] == 0) ++start;

    std::string result;
    for (int k = start; k < static_cast<int>(res.size()); ++k) {
        result += static_cast<char>('0' + res[k]);
    }
    return result;
}

int main() {
    assert(multiply("2", "3") == "6");
    assert(multiply("123", "456") == "56088");
    assert(multiply("0", "123") == "0");
    assert(multiply("9", "9") == "81");
    assert(multiply("999", "999") == "998001");
    assert(multiply("123456789", "987654321") == "121932631112635269");
    assert(multiply("100", "10") == "1000");
    std::cout << "multiply_strings: all tests passed\n";
    return 0;
}
