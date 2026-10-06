// 172. 阶乘后的零
// 见 factorial_trailing_zeroes.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int factorialTrailingZeroes(int n) {
    int count = 0;
    while (n > 0) {
        n /= 5;
        count += n;
    }
    return count;
}

int main() {
    assert(factorialTrailingZeroes(3) == 0);
    assert(factorialTrailingZeroes(5) == 1);
    assert(factorialTrailingZeroes(10) == 2);
    assert(factorialTrailingZeroes(25) == 6);
    assert(factorialTrailingZeroes(125) == 31);
    assert(factorialTrailingZeroes(0) == 0);
    assert(factorialTrailingZeroes(1) == 0);

    std::cout << "factorial_trailing_zeroes: all tests passed\n";
    return 0;
}
