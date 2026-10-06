// 461. 汉明距离
// 见 hamming_distance.py 的题目与思路说明。
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

int hammingDistance(int x, int y) {
    return hammingWeight(static_cast<std::uint32_t>(x ^ y));
}

int main() {
    assert(hammingDistance(1, 4) == 2);
    assert(hammingDistance(3, 1) == 1);
    assert(hammingDistance(0, 0) == 0);
    assert(hammingDistance(0, static_cast<int>(0xFFFFFFFFu)) == 32);
    assert(hammingDistance(7, 7) == 0);

    std::cout << "hamming_distance: all tests passed\n";
    return 0;
}
