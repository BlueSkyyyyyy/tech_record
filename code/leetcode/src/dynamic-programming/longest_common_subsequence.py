"""1143. 最长公共子序列（Longest Common Subsequence）

题目：给定两个字符串 text1 和 text2，返回它们最长公共子序列的长度。子序列不要求
连续，只要求相对顺序不变；若没有公共子序列返回 0。

思路（二维 DP：两个前缀的答案）：
    两个字符串、可以跳字符，这是「双序列 DP」的标准形状。设

        dp[i][j] = text1 的前 i 个字符与 text2 的前 j 个字符的最长公共子序列长度。

    只看两个前缀各自的最后一个字符 text1[i-1]、text2[j-1]：

    - 若相等：它们一定可以配成公共子序列的末尾，直接接在去掉这两个字符的答案后：

        dp[i][j] = dp[i-1][j-1] + 1

    - 若不等：二者不可能同时作为末尾，于是至少舍弃一个，取两种舍弃里更优的：

        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    初始化 dp[0][*] = dp[*][0] = 0（任一字符串为空，公共长度为 0）。外层 i、内层
    j 都从小到大，保证 dp[i-1][j-1]、dp[i-1][j]、dp[i][j-1] 已算好。答案 dp[m][n]。

    为什么不是 dp[i-1][j-1] 直接继承：字符不等时，跳过 text1[i-1]、跳过 text2[j-1]
    可以分别发生，所以要把这两种情况都考虑，不能只看对角。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


if __name__ == "__main__":
    assert longest_common_subsequence("abcde", "ace") == 3
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("", "abc") == 0
    assert longest_common_subsequence("bl", "yby") == 1
    print("longest_common_subsequence: all tests passed")
