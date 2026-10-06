// 201. 数字范围按位与
// 见 range_bitwise_and.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int rangeBitwiseAnd(int left, int right) {
    int shift = 0;
    while (left < right) {
        left >>= 1;
        right >>= 1;
        ++shift;
    }
    return left << shift;
}

int main() {
    assert(rangeBitwiseAnd(5, 7) == 4);
    assert(rangeBitwiseAnd(0, 0) == 0);
    assert(rangeBitwiseAnd(1, 1) == 1);
    assert(rangeBitwiseAnd(0, 1) == 0);
    assert(rangeBitwiseAnd(1, 2) == 0);
    assert(rangeBitwiseAnd(10, 10) == 10);
    assert(rangeBitwiseAnd(1, 3) == 0);
    assert(rangeBitwiseAnd(4, 7) == 4);

    std::cout << "range_bitwise_and: all tests passed\n";
    return 0;
}
