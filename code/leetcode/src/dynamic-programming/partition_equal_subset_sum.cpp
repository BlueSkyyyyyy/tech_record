// 416. 分割等和子集
// 见 partition_equal_subset_sum.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

bool canPartition(const std::vector<int> &nums) {
    int total = 0;
    for (int x : nums) total += x;
    if (total % 2 != 0) return false;
    int target = total / 2;
    std::vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int num : nums) {
        for (int i = target; i >= num; --i) {
            dp[i] = dp[i] || dp[i - num];
        }
    }
    return dp[target];
}

int main() {
    std::vector<int> a = {1, 5, 11, 5};
    std::vector<int> b = {1, 2, 3, 5};
    std::vector<int> c = {1, 1};
    std::vector<int> d = {2, 2, 2};
    std::vector<int> e = {1, 2, 5};
    assert(canPartition(a) == true);
    assert(canPartition(b) == false);
    assert(canPartition(c) == true);
    assert(canPartition(d) == false);
    assert(canPartition(e) == false);
    std::cout << "partition_equal_subset_sum: all tests passed\n";
    return 0;
}
