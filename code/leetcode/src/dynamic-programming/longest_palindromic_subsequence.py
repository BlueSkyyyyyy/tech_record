"""516. 最长回文子序列（Longest Palindromic Subsequence）

题目：给一个字符串 s，找出其中最长的回文子序列的长度。子序列不要求连续，
但必须保持字符相对顺序。

思路（区间 DP：看区间两端的字符配不配对）：
    dp[i][j] = 子串 s[i..j] 中最长回文子序列的长度。
    看区间两端：

    - s[i] == s[j]：它们可以和中间的最优解拼成一个更长的回文，
      dp[i][j] = dp[i+1][j-1] + 2；
    - s[i] != s[j]：两端不可能同时用上，丢弃其中一个，
      dp[i][j] = max(dp[i+1][j], dp[i][j-1])。

    单字符区间是长度 1 的回文，dp[i][i] = 1。
    因为 dp[i][j] 依赖 i+1 行和本行 j-1 列，所以 i 从大到小、j 从小到大遍历。

    它与最长回文子串（5，要求连续）的区别：子序列可以跳过字符，所以
    两端不等时是「丢一边取 max」，而不是「以某中心向两边扩展」。

复杂度：时间 O(n²)，空间 O(n²)。
"""


def longest_palindrome_subseq(s):
    n = len(s)
    if n == 0:
        return 0
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]


if __name__ == "__main__":
    assert longest_palindrome_subseq("bbbab") == 4
    assert longest_palindrome_subseq("cbbd") == 2
    assert longest_palindrome_subseq("a") == 1
    assert longest_palindrome_subseq("") == 0
    assert longest_palindrome_subseq("abcde") == 1
    print("longest_palindromic_subsequence: all tests passed")
