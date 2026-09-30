"""3. 无重复字符的最长子串（Longest Substring Without Repeating Characters）

题目：给定字符串 s，找出不含重复字符的**最长子串**的长度。例如 "abcabcbb" 的答案是 "abc"，长度 3。

思路：滑动窗口。用 [left, right] 维护一个「无重复字符」的窗口，right 不断右扩。
     当新字符 ch 上次出现的位置 last[ch] 落在窗口内时，说明窗口里已经有 ch，
     把 left 直接跳到 last[ch] + 1（跳过那个旧位置），窗口重新变干净。
     窗口长度 right - left + 1 的最大值即为答案。

     为什么 left 能「跳」而不是「一格一格挪」：所有以旧位置及其左侧为左端点、
     当前 right 为右端点的窗口都仍然含重复，且只会更短，所以可以直接跳过。

     为什么需要 `last[ch] >= left` 这个判断：若上次出现的位置在 left 左边，
     它已经被移出窗口了，不构成重复，此时 left 不应回退。这是最常见的易错点。

复杂度：时间 O(n)（每个字符最多被扫一次），空间 O(字符集大小)。
"""


def length_of_longest_substring(s):
    last = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best


if __name__ == "__main__":
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3
    assert length_of_longest_substring("") == 0
    assert length_of_longest_substring("dvdf") == 3
    assert length_of_longest_substring("abba") == 2
    print("longest_substring_without_repeating: all tests passed")
