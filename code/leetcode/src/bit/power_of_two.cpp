// 231. 2 的幂
// 见 power_of_two.py 的题目与思路说明。
#include <cassert>
#include <iostream>

bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int main() {
    assert(isPowerOfTwo(1) == true);
    assert(isPowerOfTwo(2) == true);
    assert(isPowerOfTwo(4) == true);
    assert(isPowerOfTwo(16) == true);
    assert(isPowerOfTwo(3) == false);
    assert(isPowerOfTwo(0) == false);
    assert(isPowerOfTwo(-16) == false);
    assert(isPowerOfTwo(1 << 30) == true);

    std::cout << "power_of_two: all tests passed\n";
    return 0;
}
