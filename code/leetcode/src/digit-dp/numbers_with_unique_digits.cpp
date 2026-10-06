// 357. 统计各位数字都不同的数字个数
// 见 numbers_with_unique_digits.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int countNumbersWithUniqueDigits(int n) {
    if (n == 0) {
        return 1;
    }
    long long total = 10;
    long long cur = 9;
    for (int length = 2; length <= n; ++length) {
        cur *= 10 - length + 1;
        total += cur;
    }
    return static_cast<int>(total);
}

int main() {
    assert(countNumbersWithUniqueDigits(0) == 1);
    assert(countNumbersWithUniqueDigits(1) == 10);
    assert(countNumbersWithUniqueDigits(2) == 91);
    assert(countNumbersWithUniqueDigits(3) == 739);
    assert(countNumbersWithUniqueDigits(11) == 8877691);
    std::cout << "numbers_with_unique_digits: all tests passed\n";
    return 0;
}
