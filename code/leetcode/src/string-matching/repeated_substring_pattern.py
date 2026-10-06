"""459. 重复的子字符串（Repeated Substring Pattern）

题目：给定非空字符串 s，判断它能否由它的某个子串重复多次构成。

思路（前缀函数判最小周期）：
    设 n = len(s)，前缀函数 lps[n-1] = L 是整串最长的相等前后缀长度。那么
    p = n - L 就是「最小可能周期」的长度。直觉：如果 s 由长度 p 的块重复而成，
    把整串往右挪 p 位，能重合的部分长度正是 n - p = L（后面的块对上前面错开的块），
    所以 L = n - p 是该种重复下前后缀的最长公共长度。

    反过来，只要 L > 0 且 n % p == 0，s 就恰好是 s[:p] 重复 n/p 次。L > 0 排除
    「没有相等前后缀」（例如 "abcd" 的 L=0，p=n，会被 n%p==0 误判成整串一个周期）。

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


def repeated_substring_pattern(s):
    n = len(s)
    lps = build_lps(s)
    period = n - lps[-1]
    return lps[-1] > 0 and n % period == 0


if __name__ == "__main__":
    assert repeated_substring_pattern("abab") is True
    assert repeated_substring_pattern("aba") is False
    assert repeated_substring_pattern("abcabcabcabc") is True
    assert repeated_substring_pattern("a") is False
    assert repeated_substring_pattern("aaaa") is True
    assert repeated_substring_pattern("abcd") is False
    assert repeated_substring_pattern("ababab") is True
    print("repeated_substring_pattern: all tests passed")
