// 11. 盛最多水的容器（对撞双指针）
// 见 container_with_most_water.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int maxArea(const std::vector<int> &height) {
    int lo = 0, hi = static_cast<int>(height.size()) - 1;
    int best = 0;
    while (lo < hi) {
        int h = std::min(height[lo], height[hi]);
        best = std::max(best, h * (hi - lo));
        if (height[lo] < height[hi])
            ++lo;
        else
            --hi;
    }
    return best;
}

int main() {
    assert(maxArea({1, 8, 6, 2, 5, 4, 8, 3, 7}) == 49);
    assert(maxArea({1, 1}) == 1);
    assert(maxArea({4, 3, 2, 1, 4}) == 16);
    assert(maxArea({1, 2, 1}) == 2);
    std::cout << "container_with_most_water: all tests passed\n";
    return 0;
}
