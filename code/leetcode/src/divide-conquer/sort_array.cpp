// 912. 排序数组（归并排序）
// 见 sort_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

void mergeSort(std::vector<int> &nums, std::vector<int> &tmp, int lo, int hi) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    mergeSort(nums, tmp, lo, mid);
    mergeSort(nums, tmp, mid, hi);

    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (nums[i] <= nums[j]) tmp[k++] = nums[i++];
        else tmp[k++] = nums[j++];
    }
    while (i < mid) tmp[k++] = nums[i++];
    while (j < hi) tmp[k++] = nums[j++];
    for (int t = lo; t < hi; ++t) nums[t] = tmp[t];
}

std::vector<int> sortArray(std::vector<int> nums) {
    std::vector<int> tmp(nums.size());
    mergeSort(nums, tmp, 0, static_cast<int>(nums.size()));
    return nums;
}

int main() {
    std::vector<int> want1 = {1, 2, 3, 5};
    assert(sortArray({5, 2, 3, 1}) == want1);

    std::vector<int> want2 = {0, 0, 1, 1, 2, 5};
    assert(sortArray({5, 1, 1, 2, 0, 0}) == want2);

    assert(sortArray({}).empty());

    std::vector<int> want3 = {1};
    assert(sortArray({1}) == want3);

    std::vector<int> want4 = {-3, -3, 0, 7, 7};
    assert(sortArray({-3, 7, -3, 0, 7}) == want4);
    std::cout << "sort_array: all tests passed\n";
    return 0;
}
