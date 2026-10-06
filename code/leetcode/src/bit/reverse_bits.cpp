// 190. 颠倒二进制位
// 见 reverse_bits.py 的题目与思路说明。
#include <cassert>
#include <cstdint>
#include <iostream>

std::uint32_t reverseBits(std::uint32_t n) {
    std::uint32_t ans = 0;
    for (int i = 0; i < 32; ++i) {
        ans = (ans << 1) | (n & 1u);
        n >>= 1;
    }
    return ans;
}

int main() {
    assert(reverseBits(0b00000010100101000001111010011100u) == 0b00111001011110000010100101000000u);
    assert(reverseBits(0u) == 0u);
    assert(reverseBits(1u) == 0x80000000u);
    assert(reverseBits(0xFFFFFFFFu) == 0xFFFFFFFFu);
    assert(reverseBits(2u) == 0x40000000u);

    std::cout << "reverse_bits: all tests passed\n";
    return 0;
}
