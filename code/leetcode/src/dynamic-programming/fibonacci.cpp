// 509. 斐波那契数
// 见 fibonacci.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int fib(int n) {
    if (n < 2) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; ++i) {
        int cur = prev2 + prev1;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

int main() {
    assert(fib(0) == 0);
    assert(fib(1) == 1);
    assert(fib(2) == 1);
    assert(fib(3) == 2);
    assert(fib(4) == 3);
    assert(fib(10) == 55);
    assert(fib(30) == 832040);
    std::cout << "fibonacci: all tests passed\n";
    return 0;
}
