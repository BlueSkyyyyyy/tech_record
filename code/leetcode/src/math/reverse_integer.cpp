// 7. 整数反转
// 见 reverse_integer.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>

int reverseInteger(int x) {
    long long rev = 0;
    while (x != 0) {
        rev = rev * 10 + x % 10;
        x /= 10;
    }
    if (rev < INT_MIN || rev > INT_MAX) {
        return 0;
    }
    return static_cast<int>(rev);
}

int main() {
    assert(reverseInteger(123) == 321);
    assert(reverseInteger(-123) == -321);
    assert(reverseInteger(120) == 21);
    assert(reverseInteger(0) == 0);
    assert(reverseInteger(7) == 7);
    assert(reverseInteger(1534236469) == 0);
    assert(reverseInteger(INT_MIN) == 0);
    assert(reverseInteger(INT_MAX) == 0);

    std::cout << "reverse_integer: all tests passed\n";
    return 0;
}
