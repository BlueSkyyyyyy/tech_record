// 154. 寻找旋转排序数组中的最小值 II（含重复）
// 见 find_minimum_in_rotated_sorted_array_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int findMinII(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right])
            left = mid + 1;
        else if (nums[mid] < nums[right])
            right = mid;
        else
            right -= 1;
    }
    return nums[left];
}

int main() {
    assert(findMinII({1, 3, 5}) == 1);
    assert(findMinII({2, 2, 2, 0, 1}) == 0);
    assert(findMinII({1, 1, 1, 0, 1}) == 0);
    assert(findMinII({1, 0, 1, 1, 1}) == 0);
    assert(findMinII({3, 3, 1, 3}) == 1);
    assert(findMinII({1}) == 1);
    assert(findMinII({1, 1}) == 1);
    assert(findMinII({2, 2, 2, 2}) == 2);
    std::cout << "find_minimum_in_rotated_sorted_array_ii: all tests passed\n";
    return 0;
}
