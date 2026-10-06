// 29. 两数相除
// 见 divide_two_integers.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>

int divideTwoIntegers(int dividend, int divisor) {
    if (dividend == INT_MIN && divisor == -1) {
        return INT_MAX;
    }
    bool negative = (dividend < 0) != (divisor < 0);
    long long a = std::llabs(static_cast<long long>(dividend));
    long long b = std::llabs(static_cast<long long>(divisor));
    long long result = 0;
    while (a >= b) {
        int shift = 0;
        while (a >= (b << (shift + 1))) {
            ++shift;
        }
        a -= b << shift;
        result += 1LL << shift;
    }
    return negative ? static_cast<int>(-result) : static_cast<int>(result);
}

int main() {
    assert(divideTwoIntegers(10, 3) == 3);
    assert(divideTwoIntegers(7, -3) == -2);
    assert(divideTwoIntegers(-7, 3) == -2);
    assert(divideTwoIntegers(-7, -3) == 2);
    assert(divideTwoIntegers(0, 1) == 0);
    assert(divideTwoIntegers(1, 1) == 1);
    assert(divideTwoIntegers(-1, -1) == 1);
    assert(divideTwoIntegers(1, 2) == 0);
    assert(divideTwoIntegers(INT_MIN, -1) == INT_MAX);
    assert(divideTwoIntegers(INT_MIN, 1) == INT_MIN);

    std::cout << "divide_two_integers: all tests passed\n";
    return 0;
}
