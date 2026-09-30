// 33. 搜索旋转排序数组
// 见 search_in_rotated_sorted_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int searchRotated(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }
    return -1;
}

int main() {
    assert(searchRotated({4, 5, 6, 7, 0, 1, 2}, 0) == 4);
    assert(searchRotated({4, 5, 6, 7, 0, 1, 2}, 3) == -1);
    assert(searchRotated({1}, 0) == -1);
    assert(searchRotated({1}, 1) == 0);
    assert(searchRotated({1, 3}, 3) == 1);
    assert(searchRotated({5, 1, 3}, 5) == 0);
    assert(searchRotated({5, 1, 3}, 3) == 2);
    assert(searchRotated({3, 1}, 1) == 1);
    std::cout << "search_in_rotated_sorted_array: all tests passed\n";
    return 0;
}
