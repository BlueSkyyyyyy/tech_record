// 233. 数字 1 的个数
// 见 number_of_digit_one.py 的题目与思路说明。
#include <cassert>
#include <iostream>

long long countDigitOne(long long n) {
    long long count = 0;
    for (long long p = 1; p <= n; p *= 10) {
        long long high = n / (p * 10);
        int cur = static_cast<int>((n / p) % 10);
        long long low = n % p;
        if (cur > 1) {
            count += (high + 1) * p;
        } else if (cur == 1) {
            count += high * p + low + 1;
        } else {
            count += high * p;
        }
    }
    return count;
}

int main() {
    assert(countDigitOne(0) == 0);
    assert(countDigitOne(13) == 6);
    assert(countDigitOne(1) == 1);
    assert(countDigitOne(10) == 2);
    assert(countDigitOne(99) == 20);
    assert(countDigitOne(100) == 21);
    assert(countDigitOne(824883294) == 767944060);
    std::cout << "number_of_digit_one: all tests passed\n";
    return 0;
}
