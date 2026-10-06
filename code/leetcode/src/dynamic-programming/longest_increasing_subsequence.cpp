// 300. 最长递增子序列
// 见 longest_increasing_subsequence.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int lengthOfLIS(const std::vector<int> &nums) {
    if (nums.empty()) return 0;
    std::vector<int> dp(nums.size(), 1);
    int best = 1;
    for (size_t i = 0; i < nums.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (nums[j] < nums[i]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
        best = std::max(best, dp[i]);
    }
    return best;
}

int main() {
    std::vector<int> a = {10, 9, 2, 5, 3, 7, 101, 18};
    std::vector<int> b = {0, 1, 0, 3, 2, 3};
    std::vector<int> c = {7, 7, 7, 7, 7};
    std::vector<int> d = {1, 2, 3, 4, 5};
    std::vector<int> e = {};
    assert(lengthOfLIS(a) == 4);
    assert(lengthOfLIS(b) == 4);
    assert(lengthOfLIS(c) == 1);
    assert(lengthOfLIS(d) == 5);
    assert(lengthOfLIS(e) == 0);
    std::cout << "longest_increasing_subsequence: all tests passed\n";
    return 0;
}
