// 35. 搜索插入位置（lower_bound）
// 见 search_insert.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int searchInsert(const std::vector<int> &nums, int target) {
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

int main() {
    assert(searchInsert({1, 3, 5, 6}, 5) == 2);
    assert(searchInsert({1, 3, 5, 6}, 2) == 1);
    assert(searchInsert({1, 3, 5, 6}, 7) == 4);
    assert(searchInsert({1, 3, 5, 6}, 0) == 0);
    assert(searchInsert({1}, 0) == 0);
    assert(searchInsert({1}, 2) == 1);
    assert(searchInsert({1}, 1) == 0);
    assert(searchInsert({}, 5) == 0);
    std::cout << "search_insert: all tests passed\n";
    return 0;
}
