"""5. 最长回文子串（Longest Palindromic Substring）

题目：给定字符串 s，找到 s 中最长的回文子串。

思路（中心扩展：枚举每一个可能的中心向两边生长）：
    回文串有一个「中心」，从中心向两边对称展开，字符始终相等。中心有两种：

    - 奇数长度回文：中心是一个字符，如 "aba" 的中心是 'b'；
    - 偶数长度回文：中心是相邻两个字符之间的空隙，如 "abba" 的中心在 'b' 和 'b' 之间。

    于是枚举所有 2n-1 个中心（n 个字符 + n-1 个空隙），各自向两边扩展到不能扩展为止，
    记录最长的那段。共 n 个奇中心 + n-1 个偶中心，对应代码里的 expand(i, i) 和
    expand(i, i+1)。

    为什么不用二维区间 DP：区间 DP 设 dp[i][j] 表示 s[i..j] 是否回文，转移是
    s[i] == s[j] 且 dp[i+1][j-1] 为真，时间同样是 O(n^2) 但要多 O(n^2) 空间。
    中心扩展时间相当、空间 O(1)，所以这里详展中心扩展，区间 DP 一句话带过。

复杂度：时间 O(n^2)（每个中心最多扩展 O(n)，共 O(n) 个中心），空间 O(1)。
"""


def longest_palindrome(s):
    if not s:
        return ""

    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    start, end = 0, 0
    for i in range(len(s)):
        l1, r1 = expand(i, i)
        l2, r2 = expand(i, i + 1)
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2
    return s[start:end + 1]


if __name__ == "__main__":
    assert longest_palindrome("babad") in ("bab", "aba")
    assert longest_palindrome("cbbd") == "bb"
    assert longest_palindrome("a") == "a"
    assert longest_palindrome("ac") in ("a", "c")
    assert longest_palindrome("") == ""
    print("longest_palindromic_substring: all tests passed")
