// 704. 二分查找（左闭右闭）
// 见 binary_search.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int search(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

int main() {
    assert(search({-1, 0, 3, 5, 9, 12}, 9) == 4);
    assert(search({-1, 0, 3, 5, 9, 12}, 2) == -1);
    assert(search({5}, 5) == 0);
    assert(search({5}, -5) == -1);
    assert(search({1, 2, 3, 4, 5}, 1) == 0);
    assert(search({1, 2, 3, 4, 5}, 5) == 4);
    assert(search({}, 1) == -1);
    std::cout << "binary_search: all tests passed\n";
    return 0;
}
