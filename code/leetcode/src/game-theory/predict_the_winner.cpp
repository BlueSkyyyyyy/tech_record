// 486. 预测赢家
// 见 predict_the_winner.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

bool predictTheWinner(std::vector<int> nums) {
    int n = nums.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        dp[i][i] = nums[i];
    }
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            dp[i][j] = std::max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1] >= 0;
}

int main() {
    std::vector<int> a1{1, 5, 2};
    std::vector<int> a2{1, 5, 233, 7};
    std::vector<int> a3{5};
    std::vector<int> a4{1, 1};
    std::vector<int> a5{2, 4, 1, 2, 7, 8};
    assert(predictTheWinner(a1) == false);
    assert(predictTheWinner(a2) == true);
    assert(predictTheWinner(a3) == true);
    assert(predictTheWinner(a4) == true);
    assert(predictTheWinner(a5) == true);

    std::cout << "predict_the_winner: all tests passed\n";
    return 0;
}
