// 746. 使用最小花费爬楼梯
// 见 min_cost_climbing_stairs.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int minCostClimbingStairs(const std::vector<int> &cost) {
    int n = static_cast<int>(cost.size());
    if (n <= 1) return 0;
    int prev2 = cost[0], prev1 = cost[1];
    for (int i = 2; i < n; ++i) {
        int cur = std::min(prev2, prev1) + cost[i];
        prev2 = prev1;
        prev1 = cur;
    }
    return std::min(prev2, prev1);
}

int main() {
    std::vector<int> a = {10, 15, 20};
    std::vector<int> b = {1, 100, 1, 1, 1, 100, 1, 1, 100, 1};
    std::vector<int> c = {0, 0, 0, 0};
    std::vector<int> d = {5, 3};
    assert(minCostClimbingStairs(a) == 15);
    assert(minCostClimbingStairs(b) == 6);
    assert(minCostClimbingStairs(c) == 0);
    assert(minCostClimbingStairs(d) == 3);
    std::cout << "min_cost_climbing_stairs: all tests passed\n";
    return 0;
}
