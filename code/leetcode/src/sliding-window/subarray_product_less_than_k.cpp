// 713. 乘积小于 K 的子数组（变长滑动窗口 + 计数）
// 见 subarray_product_less_than_k.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int numSubarrayProductLessThanK(const std::vector<int> &nums, int k) {
    if (k <= 1) return 0;
    long long prod = 1;
    int left = 0, count = 0;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        prod *= nums[right];
        while (prod >= k) {
            prod /= nums[left];
            ++left;
        }
        count += right - left + 1;
    }
    return count;
}

int main() {
    assert(numSubarrayProductLessThanK({10, 5, 2, 6}, 100) == 8);
    assert(numSubarrayProductLessThanK({1, 2, 3}, 0) == 0);
    assert(numSubarrayProductLessThanK({1, 1, 1}, 2) == 6);
    assert(numSubarrayProductLessThanK({1, 2, 3}, 1) == 0);
    assert(numSubarrayProductLessThanK({2, 3}, 6) == 2);
    std::cout << "subarray_product_less_than_k: all tests passed\n";
    return 0;
}
