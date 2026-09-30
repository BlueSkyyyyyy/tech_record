// 15. 三数之和（排序 + 固定一个数 + 对撞双指针）
// 见 three_sum.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> threeSum(std::vector<int> nums) {
    std::sort(nums.begin(), nums.end());
    int n = static_cast<int>(nums.size());
    std::vector<std::vector<int>> res;
    for (int i = 0; i < n - 2; ++i) {
        if (nums[i] > 0) break;
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int lo = i + 1, hi = n - 1;
        while (lo < hi) {
            int s = nums[i] + nums[lo] + nums[hi];
            if (s < 0) {
                ++lo;
            } else if (s > 0) {
                --hi;
            } else {
                res.push_back({nums[i], nums[lo], nums[hi]});
                ++lo;
                --hi;
                while (lo < hi && nums[lo] == nums[lo - 1]) ++lo;
                while (lo < hi && nums[hi] == nums[hi + 1]) --hi;
            }
        }
    }
    return res;
}

int main() {
    assert((threeSum({-1, 0, 1, 2, -1, -4}) ==
            std::vector<std::vector<int>>{{-1, -1, 2}, {-1, 0, 1}}));
    assert(threeSum({0, 1, 1}).empty());
    assert((threeSum({0, 0, 0}) == std::vector<std::vector<int>>{{0, 0, 0}}));
    assert(threeSum({}).empty());
    assert((threeSum({0, 0, 0, 0}) == std::vector<std::vector<int>>{{0, 0, 0}}));
    assert((threeSum({-2, 0, 0, 2, 2}) == std::vector<std::vector<int>>{{-2, 0, 2}}));
    std::cout << "three_sum: all tests passed\n";
    return 0;
}
