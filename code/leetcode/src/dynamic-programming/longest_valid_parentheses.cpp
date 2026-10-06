// 32. 最长有效括号
// 见 longest_valid_parentheses.py 的题目与思路说明。
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int longestValidParentheses(const std::string &s) {
    int n = s.size();
    std::vector<int> dp(n, 0);
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (s[i] == ')') {
            if (s[i - 1] == '(') {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (dp[i - 1] > 0) {
                int j = i - dp[i - 1] - 1;
                if (j >= 0 && s[j] == '(') {
                    dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
                }
            }
            best = std::max(best, dp[i]);
        }
    }
    return best;
}

int main() {
    assert(longestValidParentheses("(()") == 2);
    assert(longestValidParentheses(")()())") == 4);
    assert(longestValidParentheses("") == 0);
    assert(longestValidParentheses("()(())") == 6);
    assert(longestValidParentheses("(()())") == 6);
    assert(longestValidParentheses("()(()") == 2);
    assert(longestValidParentheses(")(") == 0);
    std::cout << "longest_valid_parentheses: all tests passed\n";
    return 0;
}
