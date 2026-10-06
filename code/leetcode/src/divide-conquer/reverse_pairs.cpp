// 493. 翻转对
// 见 reverse_pairs.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

int mergeSort(std::vector<int> &nums, std::vector<int> &tmp, int lo, int hi) {
    if (hi - lo <= 1) return 0;
    int mid = (lo + hi) / 2;
    int count = mergeSort(nums, tmp, lo, mid) + mergeSort(nums, tmp, mid, hi);

    int j = mid;
    for (int i = lo; i < mid; ++i) {
        while (j < hi && static_cast<long long>(nums[i]) > 2LL * nums[j]) ++j;
        count += j - mid;
    }

    int i = lo, k = lo;
    j = mid;
    while (i < mid && j < hi) {
        if (nums[i] <= nums[j]) tmp[k++] = nums[i++];
        else tmp[k++] = nums[j++];
    }
    while (i < mid) tmp[k++] = nums[i++];
    while (j < hi) tmp[k++] = nums[j++];
    for (int t = lo; t < hi; ++t) nums[t] = tmp[t];
    return count;
}

int reversePairs(std::vector<int> nums) {
    std::vector<int> tmp(nums.size());
    return mergeSort(nums, tmp, 0, static_cast<int>(nums.size()));
}

int main() {
    assert(reversePairs({1, 3, 2, 3, 1}) == 2);
    assert(reversePairs({2, 4, 3, 5, 1}) == 3);
    assert(reversePairs({1, 2, 3, 4}) == 0);
    assert(reversePairs({}) == 0);
    assert(reversePairs({1}) == 0);
    assert(reversePairs({5, 5}) == 0);
    assert(reversePairs({-1, -2}) == 1);
    assert(reversePairs({5, 4, 3, 2, 1}) == 4);
    std::cout << "reverse_pairs: all tests passed\n";
    return 0;
}
