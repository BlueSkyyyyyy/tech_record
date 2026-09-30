// 643. 子数组最大平均数 I（定长滑动窗口）
// 见 maximum_average_subarray.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

double findMaxAverage(const std::vector<int> &nums, int k) {
    long long window = 0;
    for (int i = 0; i < k; ++i) window += nums[i];
    long long best = window;
    for (int i = k; i < static_cast<int>(nums.size()); ++i) {
        window += nums[i] - nums[i - k];
        best = std::max(best, window);
    }
    return static_cast<double>(best) / k;
}

int main() {
    assert(std::abs(findMaxAverage({1, 12, -5, -6, 50, 3}, 4) - 12.75) < 1e-9);
    assert(std::abs(findMaxAverage({5}, 1) - 5.0) < 1e-9);
    assert(std::abs(findMaxAverage({0, 4, 0, 3, 2}, 1) - 4.0) < 1e-9);
    assert(std::abs(findMaxAverage({-1, -2, -3, -4}, 2) - (-1.5)) < 1e-9);
    std::cout << "maximum_average_subarray: all tests passed\n";
    return 0;
}
