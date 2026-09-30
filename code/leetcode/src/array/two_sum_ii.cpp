// 167. 两数之和 II - 输入有序数组（对撞双指针）
// 见 two_sum_ii.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

// 返回 1-indexed 的两个下标；无解时返回空 vector（题目保证有解）。
std::vector<int> twoSumSorted(const std::vector<int> &numbers, int target) {
    int lo = 0, hi = static_cast<int>(numbers.size()) - 1;
    while (lo < hi) {
        int sum = numbers[lo] + numbers[hi];
        if (sum == target) return {lo + 1, hi + 1};
        if (sum < target)
            ++lo;
        else
            --hi;
    }
    return {};
}

int main() {
    assert((twoSumSorted({2, 7, 11, 15}, 9) == std::vector<int>{1, 2}));
    assert((twoSumSorted({2, 3, 4}, 6) == std::vector<int>{1, 3}));
    assert((twoSumSorted({-1, 0}, -1) == std::vector<int>{1, 2}));
    assert((twoSumSorted({1, 2, 3, 4, 4, 9}, 8) == std::vector<int>{4, 5}));
    std::cout << "two_sum_ii: all tests passed\n";
    return 0;
}
