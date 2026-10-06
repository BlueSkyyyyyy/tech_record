// 1067. 范围内的数字计数
// 见 digit_count_in_range.py 的题目与思路说明。
#include <cassert>
#include <iostream>

long long countUpto(long long n, int d) {
    if (n <= 0) {
        return 0;
    }
    long long count = 0;
    for (long long p = 1; p <= n; p *= 10) {
        long long high = n / (p * 10);
        int cur = static_cast<int>((n / p) % 10);
        long long low = n % p;
        if (d != 0) {
            if (cur > d) {
                count += (high + 1) * p;
            } else if (cur == d) {
                count += high * p + low + 1;
            } else {
                count += high * p;
            }
        } else if (high > 0) {
            if (cur == 0) {
                count += (high - 1) * p + low + 1;
            } else {
                count += high * p;
            }
        }
    }
    return count;
}

int digitCountInRange(int d, int low, int high) {
    return static_cast<int>(countUpto(high, d) - countUpto(low - 1, d));
}

int main() {
    assert(digitCountInRange(1, 1, 13) == 6);
    assert(digitCountInRange(3, 100, 250) == 35);
    assert(digitCountInRange(0, 1, 9) == 0);
    assert(digitCountInRange(0, 1, 99) == 9);
    assert(digitCountInRange(0, 1, 100) == 11);
    assert(digitCountInRange(2, 1, 22) == 6);
    assert(digitCountInRange(9, 1, 1000000) == 600000);
    std::cout << "digit_count_in_range: all tests passed\n";
    return 0;
}
