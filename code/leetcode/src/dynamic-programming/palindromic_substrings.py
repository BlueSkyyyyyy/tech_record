"""647. 回文子串（Palindromic Substrings）

题目：给一个字符串 s，统计它有多少个回文子串（不同位置算不同的子串）。

思路（布尔区间 DP：从小区间推大区间）：
    dp[i][j] 表示 s[i..j] 是否为回文。判定分两步：

        s[i] == s[j] 且 (区间长度 <= 2 或 dp[i+1][j-1] 为真)

    也就是说，只要两端字符相同，再看中间那段（长度不足 2 时中间为空，
    天然回文）。每得到一个 True 就计数 +1。

    遍历顺序：dp[i][j] 依赖 dp[i+1][j-1]（下一行、左一列），
    所以 i 从大到小、j 从小到大，保证内层已经算好。

    另一条常见路线是「中心扩展」（枚举 2n-1 个中心向两边长，O(1) 空间），
    与最长回文子串（5）同款；这里用区间 DP，是为了和 516 共用一套填表框架。

复杂度：时间 O(n²)，空间 O(n²)（中心扩展可做到 O(1)）。
"""


def count_substrings(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                dp[i][j] = True
                count += 1
    return count


if __name__ == "__main__":
    assert count_substrings("abc") == 3
    assert count_substrings("aaa") == 6
    assert count_substrings("a") == 1
    assert count_substrings("") == 0
    assert count_substrings("abba") == 6
    print("palindromic_substrings: all tests passed")
