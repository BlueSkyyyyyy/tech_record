// 34. 在排序数组中查找元素的第一个和最后一个位置（lower_bound + upper_bound）
// 见 find_first_and_last.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int lowerBound(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}

int upperBound(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] <= target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}

std::vector<int> searchRange(const std::vector<int> &nums, int target) {
    int first = lowerBound(nums, target);
    if (first == static_cast<int>(nums.size()) || nums[first] != target)
        return {-1, -1};
    int last = upperBound(nums, target) - 1;
    return {first, last};
}

int main() {
    std::vector<int> want1{-1, -1};
    std::vector<int> want2{0, 0};
    std::vector<int> want3{0, 3};
    std::vector<int> want4{1, 1};
    std::vector<int> want5{3, 4};
    assert(searchRange({5, 7, 7, 8, 8, 10}, 8) == want5);
    assert(searchRange({5, 7, 7, 8, 8, 10}, 6) == want1);
    assert(searchRange({}, 0) == want1);
    assert(searchRange({1}, 1) == want2);
    assert(searchRange({1}, 0) == want1);
    assert(searchRange({2, 2, 2, 2}, 2) == want3);
    assert(searchRange({1, 2, 3}, 2) == want4);
    assert(searchRange({1, 2, 3}, 4) == want1);
    std::cout << "find_first_and_last: all tests passed\n";
    return 0;
}
