// 70. 爬楼梯
// 见 climbing_stairs.py 的题目与思路说明。
#include <cassert>
#include <iostream>

int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; ++i) {
        int cur = prev2 + prev1;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

int main() {
    assert(climbStairs(1) == 1);
    assert(climbStairs(2) == 2);
    assert(climbStairs(3) == 3);
    assert(climbStairs(4) == 5);
    assert(climbStairs(5) == 8);
    assert(climbStairs(10) == 89);
    std::cout << "climbing_stairs: all tests passed\n";
    return 0;
}
