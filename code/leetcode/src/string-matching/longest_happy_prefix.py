"""1392. 最长快乐前缀（Longest Happy Prefix）

题目：「快乐前缀」是既是原串的非空真前缀、又是它的后缀的字符串。给定字符串 s，
返回它的最长快乐前缀；不存在则返回空串。

思路（前缀函数一步到位）：
    KMP 的前缀函数 lps[i] 的定义正是「s[0..i] 里最长的相等前后缀长度」。整串的
    最长快乐前缀，就是 lps[n-1] 对应的那段前缀 s[:lps[n-1]]（lps 天然小于 i+1，
    取到的必定是真前缀）。构造 lps 后直接切片即可，相当于把 KMP 的中间产物当答案。

复杂度：时间 O(n)，空间 O(n)。
"""


def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def longest_prefix(s):
    if not s:
        return ""
    return s[: build_lps(s)[-1]]


if __name__ == "__main__":
    assert longest_prefix("level") == "l"
    assert longest_prefix("ababab") == "abab"
    assert longest_prefix("leetcodeleet") == "leet"
    assert longest_prefix("a") == ""
    assert longest_prefix("abcd") == ""
    assert longest_prefix("aaaa") == "aaa"
    print("longest_prefix: all tests passed")
