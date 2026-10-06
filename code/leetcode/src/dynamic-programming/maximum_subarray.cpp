// 53. 最大子数组和
// 见 maximum_subarray.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int maxSubArray(const std::vector<int> &nums) {
    int best = nums[0], cur = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        cur = std::max(nums[i], cur + nums[i]);
        best = std::max(best, cur);
    }
    return best;
}

int main() {
    std::vector<int> a = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    std::vector<int> b = {1};
    std::vector<int> c = {5, 4, -1, 7, 8};
    std::vector<int> d = {-1};
    std::vector<int> e = {-3, -1, -2};
    assert(maxSubArray(a) == 6);
    assert(maxSubArray(b) == 1);
    assert(maxSubArray(c) == 23);
    assert(maxSubArray(d) == -1);
    assert(maxSubArray(e) == -1);
    std::cout << "maximum_subarray: all tests passed\n";
    return 0;
}
