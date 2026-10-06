// 50. Pow(x, n)（快速幂）
// 见 pow_x_n.py 的题目与思路说明。
#include <cassert>
#include <iostream>

double power(double base, long long exp) {
    if (exp == 0) return 1.0;
    double half = power(base, exp / 2);
    if (exp % 2 == 0) return half * half;
    return half * half * base;
}

double myPow(double x, long long n) {
    if (n < 0) {
        x = 1.0 / x;
        n = -n;
    }
    return power(x, n);
}

int main() {
    assert(myPow(2.0, 10) == 1024.0);
    assert(myPow(2.0, 0) == 1.0);
    assert(myPow(2.0, -2) == 0.25);
    assert(myPow(0.5, -2) == 4.0);
    assert(myPow(-2.0, 3) == -8.0);
    assert(myPow(1.0, -2147483648LL) == 1.0);
    std::cout << "pow_x_n: all tests passed\n";
    return 0;
}
