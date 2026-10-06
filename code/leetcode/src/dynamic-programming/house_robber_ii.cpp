// 213. 打家劫舍 II
// 见 house_robber_ii.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int robRange(const std::vector<int> &nums, int lo, int hi) {
    int prev2 = 0, prev1 = 0;
    for (int i = lo; i < hi; ++i) {
        int cur = std::max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

int rob(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    return std::max(robRange(nums, 0, n - 1), robRange(nums, 1, n));
}

int main() {
    std::vector<int> a = {2, 3, 2};
    std::vector<int> b = {1, 2, 3, 1};
    std::vector<int> c = {1, 2, 3};
    std::vector<int> d = {5};
    std::vector<int> e = {};
    std::vector<int> f = {1, 2, 1, 1};
    assert(rob(a) == 3);
    assert(rob(b) == 4);
    assert(rob(c) == 3);
    assert(rob(d) == 5);
    assert(rob(e) == 0);
    assert(rob(f) == 3);
    std::cout << "house_robber_ii: all tests passed\n";
    return 0;
}
