// 1218. 最长定差子序列（DP 用哈希表 O(1) 找前驱）
// 见 longest_arithmetic_subsequence.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

int longestSubsequence(const std::vector<int> &arr, int difference) {
    std::unordered_map<int, int> dp;
    int best = 0;
    for (int x : arr) {
        auto it = dp.find(x - difference);
        int len = (it == dp.end() ? 0 : it->second) + 1;
        dp[x] = len;
        best = std::max(best, len);
    }
    return best;
}

int main() {
    assert(longestSubsequence({1, 2, 3, 4}, 1) == 4);
    assert(longestSubsequence({1, 3, 5, 7}, 1) == 1);
    assert(longestSubsequence({1, 5, 7, 8, 5, 3, 4, 2, 1}, -2) == 4);
    assert(longestSubsequence({7}, 0) == 1);
    std::cout << "longest_arithmetic_subsequence: all tests passed\n";
    return 0;
}
