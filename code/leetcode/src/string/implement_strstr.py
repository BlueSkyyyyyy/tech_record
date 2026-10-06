"""28. 找出字符串中第一个匹配项的下标（Implement strStr / KMP）

题目：给定两个字符串 haystack 与 needle，返回 needle 在 haystack 中第一次出现的下标；
若不存在返回 -1。约定 needle 为空串时返回 0。

思路（KMP）：
    暴力做法每次失配都把 needle 的下标退回 0、haystack 的下标回退，最坏 O(n*m)。
    KMP 的关键观察是：已经匹配上的那一段本身含有信息——它的某个「既是前缀又是后缀」的
    部分可以复用，失配时不必从头再来。于是先对 needle 求一个前缀函数 lps（longest
    proper prefix which is also suffix）：

        lps[i] = needle[0..i] 中「最长的、既是真前缀又是真后缀」的长度。

    例如 needle = "ababaca" 的 lps 是 [0,0,1,2,3,0,1]。有了它，当 needle[k] 与文本失配
    时，我们把 k 退到 lps[k-1]，这表示「前面这段的公共前后缀仍然匹配，直接从这里继续比较」。

    匹配阶段让文本指针 i 单向前进、永不回退；k 表示当前 needle 已匹配的长度。每当匹配满
    整个 needle（k == m），此时的起点就是 i - m + 1。

    为什么这样是线性：k 每次最多 +1（匹配时），失配时按 lps 回退，而回退的步数总会被之前
    「增长」的次数抵消，均摊下来 k 的总变化是 O(n)，所以时间 O(n + m)。

    也可用 Python 的 str.find 一行搞定，但本题正是 KMP 的入门模板，值得手写。

复杂度：时间 O(n + m)，空间 O(m)（前缀函数数组）。
"""


def str_str(haystack, needle):
    if needle == "":
        return 0
    n, m = len(haystack), len(needle)

    lps = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and needle[i] != needle[k]:
            k = lps[k - 1]
        if needle[i] == needle[k]:
            k += 1
        lps[i] = k

    k = 0
    for i in range(n):
        while k > 0 and haystack[i] != needle[k]:
            k = lps[k - 1]
        if haystack[i] == needle[k]:
            k += 1
            if k == m:
                return i - m + 1
    return -1


if __name__ == "__main__":
    assert str_str("sadbutsad", "sad") == 0
    assert str_str("leetcode", "leeto") == -1
    assert str_str("hello", "") == 0
    assert str_str("a", "a") == 0
    assert str_str("abcabcabd", "abcabd") == 3
    assert str_str("aaaaa", "bba") == -1
    assert str_str("mississippi", "issip") == 4

    print("implement_strstr: all tests passed")
