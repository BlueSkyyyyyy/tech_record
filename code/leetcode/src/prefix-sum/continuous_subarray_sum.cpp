// 523. 连续的子数组和（同余前缀和 + 最早下标）
// 见 continuous_subarray_sum.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

bool checkSubarraySum(const std::vector<int> &nums, int k) {
    std::unordered_map<int, int> first;
    first[0] = -1;
    long long prefix = 0;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        prefix = ((prefix + nums[i]) % k + k) % k;
        auto it = first.find(static_cast<int>(prefix));
        if (it != first.end()) {
            if (i - it->second >= 2) return true;
        } else {
            first[static_cast<int>(prefix)] = i;
        }
    }
    return false;
}

int main() {
    assert(checkSubarraySum({23, 2, 4, 6, 7}, 6) == true);
    assert(checkSubarraySum({23, 2, 6, 4, 7}, 6) == true);
    assert(checkSubarraySum({23, 2, 6, 4, 7}, 13) == false);
    assert(checkSubarraySum({5, 0, 0, 0}, 3) == true);
    assert(checkSubarraySum({1, 0}, 2) == false);
    std::cout << "continuous_subarray_sum: all tests passed\n";
    return 0;
}
