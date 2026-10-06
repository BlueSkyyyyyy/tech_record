"""28. 找出字符串中第一个匹配项的下标（Implement strStr）· KMP

题目：给定两个字符串 haystack 和 needle，返回 needle 在 haystack 中第一次出现的
下标；不存在则返回 -1。needle 为空时返回 0。

思路（KMP：失配时不回退主串指针）：
    暴力匹配在某个位置失配后，会把主串指针退回去、从下一个位置重新比，最坏 O(n·m)。
    KMP 的关键观察是：已经匹配上的那一段本身携带信息——它的某个前缀可能正好是它的
    后缀。失配时，主串指针不动，只把模式串指针回退到「最长相等前后缀」的长度处继续比。

    这个「回退到哪」由模式串自己的前缀函数 lps 给出：lps[i] 表示 needle[0..i] 这一段
    里，最长的「既是前缀又是后缀」的真子串长度。构造 lps 时，用两个指针 i、length：
    若 needle[i] == needle[length]，说明前后缀又能延长一位；否则 length 回退到
    lps[length-1] 继续尝试，实在不行就置 lps[i]=0。

    匹配阶段同理：needle[j] 与 haystack[i] 相等就 j += 1；失配且 j > 0 就令
    j = lps[j-1]（主串 i 不回退）；j == m 时说明整段匹配成功，起始下标是 i - m + 1。

复杂度：时间 O(n + m)，空间 O(m)（lps 数组）。
"""


def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0  # 当前最长相等前后缀的长度
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]  # 前后缀接不上，退而求其次
        else:
            lps[i] = 0
            i += 1
    return lps


def str_str(haystack, needle):
    if needle == "":
        return 0
    lps = build_lps(needle)
    m = len(needle)
    j = 0  # needle 上已经匹配的长度
    for i in range(len(haystack)):
        while j > 0 and haystack[i] != needle[j]:
            j = lps[j - 1]  # 主串指针 i 不回退
        if haystack[i] == needle[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1


if __name__ == "__main__":
    assert build_lps("ababaca") == [0, 0, 1, 2, 3, 0, 1]
    assert str_str("sadbutsad", "sad") == 0
    assert str_str("leetcode", "leeto") == -1
    assert str_str("hello", "") == 0
    assert str_str("aabaabaaa", "aabaaa") == 3
    assert str_str("mississippi", "issip") == 4
    assert str_str("abc", "abcd") == -1
    print("str_str: all tests passed")
