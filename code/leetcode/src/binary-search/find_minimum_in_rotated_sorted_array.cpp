// 153. 寻找旋转排序数组中的最小值
// 见 find_minimum_in_rotated_sorted_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int findMin(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right])
            left = mid + 1;
        else
            right = mid;
    }
    return nums[left];
}

int main() {
    assert(findMin({3, 4, 5, 1, 2}) == 1);
    assert(findMin({4, 5, 6, 7, 0, 1, 2}) == 0);
    assert(findMin({11, 13, 15, 17}) == 11);
    assert(findMin({1}) == 1);
    assert(findMin({2, 1}) == 1);
    assert(findMin({1, 2}) == 1);
    assert(findMin({5, 1, 2, 3, 4}) == 1);
    std::cout << "find_minimum_in_rotated_sorted_array: all tests passed\n";
    return 0;
}
