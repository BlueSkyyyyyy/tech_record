// 209. 长度最小的子数组（滑动窗口 + 收缩左端）
// 见 minimum_size_subarray_sum.py 的题目与思路说明。
#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

int minSubArrayLen(int target, const std::vector<int> &nums) {
    int left = 0, total = 0, best = INT_MAX;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        total += nums[right];
        while (total >= target) {
            if (right - left + 1 < best) best = right - left + 1;
            total -= nums[left];
            ++left;
        }
    }
    return best == INT_MAX ? 0 : best;
}

int main() {
    assert(minSubArrayLen(7, {2, 3, 1, 2, 4, 3}) == 2);
    assert(minSubArrayLen(4, {1, 4, 4}) == 1);
    assert(minSubArrayLen(11, {1, 1, 1, 1, 1, 1, 1, 1}) == 0);
    assert(minSubArrayLen(15, {5, 1, 3, 5, 10, 7, 4, 9, 2, 8}) == 2);
    assert(minSubArrayLen(1, {1}) == 1);
    std::cout << "minimum_size_subarray_sum: all tests passed\n";
    return 0;
}
