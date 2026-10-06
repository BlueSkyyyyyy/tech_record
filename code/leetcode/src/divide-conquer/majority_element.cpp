// 169. 多数元素
// 见 majority_element.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int countInRange(const std::vector<int> &nums, int lo, int hi, int target) {
    int count = 0;
    for (int i = lo; i < hi; ++i) {
        if (nums[i] == target) ++count;
    }
    return count;
}

int majority(const std::vector<int> &nums, int lo, int hi) {
    if (hi - lo == 1) return nums[lo];
    int mid = (lo + hi) / 2;
    int left = majority(nums, lo, mid);
    int right = majority(nums, mid, hi);
    if (left == right) return left;
    int leftCount = countInRange(nums, lo, hi, left);
    int rightCount = countInRange(nums, lo, hi, right);
    return leftCount > rightCount ? left : right;
}

int majorityElement(const std::vector<int> &nums) {
    return majority(nums, 0, static_cast<int>(nums.size()));
}

int main() {
    std::vector<int> a = {3, 2, 3};
    assert(majorityElement(a) == 3);
    std::vector<int> b = {2, 2, 1, 1, 1, 2, 2};
    assert(majorityElement(b) == 2);
    std::vector<int> c = {1};
    assert(majorityElement(c) == 1);
    std::vector<int> d = {1, 1, 2};
    assert(majorityElement(d) == 1);
    std::vector<int> e = {1, 2, 1};
    assert(majorityElement(e) == 1);
    std::cout << "majority_element: all tests passed\n";
    return 0;
}
