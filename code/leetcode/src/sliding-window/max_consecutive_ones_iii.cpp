// 1004. 最大连续 1 的个数 III（变长滑动窗口）
// 见 max_consecutive_ones_iii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int longestOnes(const std::vector<int> &nums, int k) {
    int left = 0, zeros = 0, best = 0;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        if (nums[right] == 0) ++zeros;
        while (zeros > k) {
            if (nums[left] == 0) --zeros;
            ++left;
        }
        if (right - left + 1 > best) best = right - left + 1;
    }
    return best;
}

int main() {
    assert(longestOnes({1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0}, 2) == 6);
    assert(longestOnes({0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1}, 3) == 10);
    assert(longestOnes({1, 1, 1}, 0) == 3);
    assert(longestOnes({0, 0, 0}, 0) == 0);
    assert(longestOnes({0, 0, 0}, 3) == 3);
    std::cout << "max_consecutive_ones_iii: all tests passed\n";
    return 0;
}
