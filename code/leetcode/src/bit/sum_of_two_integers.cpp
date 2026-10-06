// 371. 两整数之和
// 见 sum_of_two_integers.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int getSum(int a, int b) {
    while (b != 0) {
        unsigned carry = static_cast<unsigned>(a & b) << 1;
        a = a ^ b;
        b = static_cast<int>(carry);
    }
    return a;
}

int main() {
    assert(getSum(1, 2) == 3);
    assert(getSum(2, 3) == 5);
    assert(getSum(-1, 1) == 0);
    assert(getSum(-2, 3) == 1);
    assert(getSum(-5, -7) == -12);
    assert(getSum(0, 0) == 0);
    assert(getSum(123, 456) == 579);

    std::cout << "sum_of_two_integers: all tests passed\n";
    return 0;
}
