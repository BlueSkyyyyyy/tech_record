// 724. 寻找数组的中心下标（前缀和）
// 见 find_pivot_index.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int pivotIndex(const std::vector<int> &nums) {
    long long total = std::accumulate(nums.begin(), nums.end(), 0LL);
    long long left = 0;
    for (std::size_t i = 0; i < nums.size(); ++i) {
        if (left == total - left - nums[i]) return static_cast<int>(i);
        left += nums[i];
    }
    return -1;
}

int main() {
    assert(pivotIndex({1, 7, 3, 6, 5, 6}) == 3);
    assert(pivotIndex({1, 2, 3}) == -1);
    assert(pivotIndex({2, 1, -1}) == 0);
    assert(pivotIndex({0, 0, 0}) == 0);
    assert(pivotIndex({1}) == 0);
    assert(pivotIndex({-1, -1, -1, -1, -1, 0}) == 2);
    std::cout << "find_pivot_index: all tests passed\n";
    return 0;
}
