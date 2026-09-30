// 69. x 的平方根（二分答案）
// 见 sqrtx.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int mySqrt(int x) {
    int left = 0, right = x;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (1LL * mid * mid <= x)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left - 1;
}

int main() {
    assert(mySqrt(0) == 0);
    assert(mySqrt(1) == 1);
    assert(mySqrt(4) == 2);
    assert(mySqrt(8) == 2);
    assert(mySqrt(9) == 3);
    assert(mySqrt(15) == 3);
    assert(mySqrt(16) == 4);
    assert(mySqrt(2147395599) == 46339);
    std::cout << "sqrtx: all tests passed\n";
    return 0;
}
