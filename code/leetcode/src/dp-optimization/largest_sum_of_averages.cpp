// 813. 最大平均值和的分组（DP + 前缀和）
// 见 largest_sum_of_averages.py 的题目与思路说明。
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

double largestSumOfAverages(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<double> pre(n + 1, 0.0);
    for (int i = 0; i < n; ++i) pre[i + 1] = pre[i] + nums[i];

    std::vector<std::vector<double>> dp(k + 1, std::vector<double>(n + 1, 0.0));
    for (int i = 1; i <= n; ++i) dp[1][i] = pre[i] / i;
    for (int j = 2; j <= k; ++j) {
        for (int i = j; i <= n; ++i) {
            double best = 0.0;
            for (int m = j - 1; m < i; ++m) {
                double cand = dp[j - 1][m] + (pre[i] - pre[m]) / (i - m);
                if (cand > best) best = cand;
            }
            dp[j][i] = best;
        }
    }
    return dp[k][n];
}

int main() {
    assert(std::fabs(largestSumOfAverages({9, 1, 2, 3, 9}, 3) - 20.0) < 1e-9);
    assert(std::fabs(largestSumOfAverages({1, 2, 3, 4, 5, 6, 7}, 4) - 20.5) < 1e-9);
    assert(std::fabs(largestSumOfAverages({5}, 1) - 5.0) < 1e-9);
    std::cout << "largest_sum_of_averages: all tests passed\n";
    return 0;
}
