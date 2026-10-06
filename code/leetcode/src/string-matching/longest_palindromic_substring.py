"""5. 最长回文子串（Longest Palindromic Substring）· Manacher

题目：给定字符串 s，返回 s 中最长的回文子串。

思路（Manacher：把回文半径用起来）：
    「从中心往两边扩」能找出以每个位置为中心的最长回文，但它对每个中心都从头扩，最坏
    O(n²)。Manacher 的加速点在于：回文之间会「互相利用」。维护当前已探明、向右延伸最远
    的回文 [center, right]，当处理到它内部的位置 i 时，i 关于 center 的对称点
    2·center - i 的回文半径可以拿来用，但最多只能用到边界 right 处，即
        p[i] = min(right - i, p[2 * center - i])
    再从这个下界继续往两边扩，扩完若越过 right 就更新 center/right。

    为了统一奇偶长度，先把 s 插成 "a#b#c" 形式（每个字符间和两端加 '#'），并在首尾放
    哨兵 '^' '$' 免去边界判断。此时所有回文都变成奇数长度，p[i] 表示以 t[i] 为中心能
    向两边扩多少对，p[i] 的数值恰好等于原串中该回文的长度。

    最后取 p 的最大值，反推回原串的起点：原串起点 = (中心下标 - 半径) // 2。

复杂度：时间 O(n)，空间 O(n)。
"""


def longest_palindrome(s):
    if len(s) <= 1:
        return s
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(1, n - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2
    return s[start : start + max_len]


if __name__ == "__main__":
    assert longest_palindrome("babad") in ("bab", "aba")
    assert longest_palindrome("cbbd") == "bb"
    assert longest_palindrome("a") == "a"
    assert longest_palindrome("") == ""
    assert longest_palindrome("abb") == "bb"
    assert longest_palindrome("abccba") == "abccba"
    assert longest_palindrome("forgeeksskeegfor") == "geeksskeeg"
    print("longest_palindrome: all tests passed")
