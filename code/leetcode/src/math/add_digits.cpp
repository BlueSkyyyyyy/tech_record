// 258. 各位相加
// 见 add_digits.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int addDigits(int num) {
    if (num == 0) {
        return 0;
    }
    return 1 + (num - 1) % 9;
}

int main() {
    assert(addDigits(38) == 2);
    assert(addDigits(0) == 0);
    assert(addDigits(1) == 1);
    assert(addDigits(9) == 9);
    assert(addDigits(18) == 9);
    assert(addDigits(10) == 1);
    assert(addDigits(99) == 9);
    assert(addDigits(999) == 9);
    assert(addDigits(12345) == 6);

    std::cout << "add_digits: all tests passed\n";
    return 0;
}
