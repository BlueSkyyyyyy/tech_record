// 718. 最长重复子数组
// 见 maximum_length_of_repeated_subarray.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int findLength(const std::vector<int> &nums1, const std::vector<int> &nums2) {
    int m = static_cast<int>(nums1.size());
    int n = static_cast<int>(nums2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                best = std::max(best, dp[i][j]);
            }
        }
    }
    return best;
}

int main() {
    std::vector<int> a1 = {1, 2, 3, 2, 1}, a2 = {3, 2, 1, 4, 7};
    std::vector<int> b1 = {0, 0, 0, 0, 0}, b2 = {0, 0, 0, 0, 0};
    std::vector<int> c1 = {1, 2, 3}, c2 = {4, 5, 6};
    std::vector<int> d1 = {}, d2 = {1, 2};
    std::vector<int> e1 = {5}, e2 = {5};
    assert(findLength(a1, a2) == 3);
    assert(findLength(b1, b2) == 5);
    assert(findLength(c1, c2) == 0);
    assert(findLength(d1, d2) == 0);
    assert(findLength(e1, e2) == 1);
    std::cout << "maximum_length_of_repeated_subarray: all tests passed\n";
    return 0;
}
