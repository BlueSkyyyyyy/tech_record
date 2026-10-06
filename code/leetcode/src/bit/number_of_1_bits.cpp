// 191. 位 1 的个数
// 见 number_of_1_bits.py 的题目与思路说明。
#include <cassert>
#include <cstdint>
#include <iostream>

int hammingWeight(std::uint32_t n) {
    int count = 0;
    while (n) {
        n &= n - 1;
        ++count;
    }
    return count;
}

int main() {
    assert(hammingWeight(0) == 0);
    assert(hammingWeight(1) == 1);
    assert(hammingWeight(2) == 1);
    assert(hammingWeight(3) == 2);
    assert(hammingWeight(11) == 3);
    assert(hammingWeight(128) == 1);
    assert(hammingWeight(255) == 8);
    assert(hammingWeight(0xFFFFFFFFu) == 32);

    std::cout << "number_of_1_bits: all tests passed\n";
    return 0;
}
