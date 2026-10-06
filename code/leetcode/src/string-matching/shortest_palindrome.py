"""214. 最短回文串（Shortest Palindrome）

题目：给定字符串 s，可以在 s 前面添加字符使其变成回文串，返回这样构造出的最短回文串。

思路（KMP 找「最长回文前缀」）：
    最终答案一定形如「补一段 + s」，而补的那段是 s 的一个后缀的逆序。要让总长度最短，
    就要让 s 里已经有尽可能长的前缀本身是回文——因为这段回文前缀可以原样当答案的中段，
    只需把剩下的后缀反过来补在最前面。

    问题转化为：s 的「最长回文前缀」有多长？把 s 和一个分隔符、再和 s 的反转拼起来：
        combined = s + '#' + reverse(s)
    分隔符 '#' 不在 s 中出现，保证匹配不会跨过边界。此时 combined 的最长相等前后缀长度
    lps[-1] = k，恰好就是 s 的最长回文前缀长度。于是把 s[k:] 反转后接到前面即可。

    为什么 lps[-1] 就是回文前缀长度：combined 的一个后缀落在 reverse(s) 部分，是 s 的
    前缀的逆；它同时也作为前缀出现在 s 的开头。两者相等 ⟺ s 的开头这段是回文。

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


def shortest_palindrome(s):
    if len(s) <= 1:
        return s
    combined = s + "#" + s[::-1]
    k = build_lps(combined)[-1]
    return s[k:][::-1] + s


if __name__ == "__main__":
    assert shortest_palindrome("aacecaaa") == "aaacecaaa"
    assert shortest_palindrome("abcd") == "dcbabcd"
    assert shortest_palindrome("a") == "a"
    assert shortest_palindrome("") == ""
    assert shortest_palindrome("aba") == "aba"
    print("shortest_palindrome: all tests passed")
