// 162. 寻找峰值
// 见 find_peak_element.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int findPeakElement(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[mid + 1])
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}

int main() {
    assert(findPeakElement({1, 2, 3, 1}) == 2);
    int p1 = findPeakElement({1, 2, 1, 3, 5, 6, 4});
    assert(p1 == 1 || p1 == 5);
    assert(findPeakElement({1}) == 0);
    assert(findPeakElement({1, 2}) == 1);
    assert(findPeakElement({2, 1}) == 0);
    assert(findPeakElement({3, 2, 1}) == 0);
    assert(findPeakElement({1, 2, 3}) == 2);
    std::cout << "find_peak_element: all tests passed\n";
    return 0;
}
