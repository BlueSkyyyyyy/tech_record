// 1. 两数之和（哈希表）
// 见 two_sum.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

// 返回两个下标（任意顺序），无解时返回空 vector（题目保证有解）。
std::vector<int> twoSum(const std::vector<int> &nums, int target) {
    std::unordered_map<int, int> seen;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        int need = target - nums[i];
        auto it = seen.find(need);
        if (it != seen.end()) return {it->second, i};
        seen[nums[i]] = i;
    }
    return {};
}

int main() {
    auto check = [](std::vector<int> got, std::vector<int> want) {
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        return got == want;
    };
    assert(check(twoSum({2, 7, 11, 15}, 9), {0, 1}));
    assert(check(twoSum({3, 2, 4}, 6), {1, 2}));
    assert(check(twoSum({3, 3}, 6), {0, 1}));
    assert(check(twoSum({-3, 4, 3, 90}, 0), {0, 2}));
    std::cout << "two_sum: all tests passed\n";
    return 0;
}
