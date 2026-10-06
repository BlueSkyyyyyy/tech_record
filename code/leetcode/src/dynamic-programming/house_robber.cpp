// 198. 打家劫舍
// 见 house_robber.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int rob(const std::vector<int> &nums) {
    int prev2 = 0, prev1 = 0;
    for (int x : nums) {
        int cur = std::max(prev1, prev2 + x);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

int main() {
    std::vector<int> a = {1, 2, 3, 1};
    std::vector<int> b = {2, 7, 9, 3, 1};
    std::vector<int> c = {};
    std::vector<int> d = {5};
    std::vector<int> e = {2, 1, 1, 2};
    assert(rob(a) == 4);
    assert(rob(b) == 12);
    assert(rob(c) == 0);
    assert(rob(d) == 5);
    assert(rob(e) == 4);
    std::cout << "house_robber: all tests passed\n";
    return 0;
}
