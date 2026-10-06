// 53. 最大子数组和（分治解）
// 见 maximum_subarray.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int solve(const std::vector<int> &nums, int lo, int hi) {
    if (hi - lo == 1) return nums[lo];
    int mid = (lo + hi) / 2;
    int leftBest = solve(nums, lo, mid);
    int rightBest = solve(nums, mid, hi);

    int best = INT_MIN, total = 0;
    for (int i = mid - 1; i >= lo; --i) {
        total += nums[i];
        best = std::max(best, total);
    }
    int leftCross = best;

    best = INT_MIN, total = 0;
    for (int i = mid; i < hi; ++i) {
        total += nums[i];
        best = std::max(best, total);
    }
    int rightCross = best;

    return std::max({leftBest, rightBest, leftCross + rightCross});
}

int maxSubArray(const std::vector<int> &nums) {
    return solve(nums, 0, static_cast<int>(nums.size()));
}

int main() {
    std::vector<int> a = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    assert(maxSubArray(a) == 6);
    std::vector<int> b = {1};
    assert(maxSubArray(b) == 1);
    std::vector<int> c = {5, 4, -1, 7, 8};
    assert(maxSubArray(c) == 23);
    std::vector<int> d = {-3, -1, -2};
    assert(maxSubArray(d) == -1);
    std::vector<int> e = {-1, -2};
    assert(maxSubArray(e) == -1);
    std::cout << "maximum_subarray: all tests passed\n";
    return 0;
}
