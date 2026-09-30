"""424. 替换后的最长重复字符（Longest Repeating Character Replacement）

题目：给定只含大写字母的字符串 s 和整数 k，最多替换 k 个字符，求能得到的最长重复字符
      子串的长度。例如 s = "AABABBA"、k = 1，答案是 4（把中间的 "B" 换成 "A" 得 "AAAA"）。

思路：变长滑动窗口。窗口内「让所有字符都变成同一个字符」所需替换次数 = 窗口长度 - 窗口内
      出现次数最多的字符的频次。只要这个值 <= k，窗口就合法。

      于是右端一路右扩，维护窗口内各字符计数以及最大的频次 max_freq。当
      (right - left + 1) - max_freq > k 时，说明窗口太长、替换不过来了，把左端右移一格。

      为什么这里可以用 if 而不是 while：窗口长度最多只会比上一轮大 1，所以当条件被破坏时
      左端最多右移一格就能恢复；而且我们要的是「最长」，窗口长度从头到尾不减，用历史最大
      长度作答案即可。用 while 也对，只是多做无用的收缩。

      为什么 max_freq 缩小后不用回退更新：max_freq 记的是历史最大值，即便它对应的字符
      已经不在窗口里，用偏大的 max_freq 只会让条件更宽松、窗口更大。我们要求的是最大长度，
      偏宽松不会得到「错误但更大」的答案（更大必来自真实合法窗口），所以可不更新。要理解这点
      最好在草稿纸上走一遍 "AABABBA"。

复杂度：时间 O(n)，空间 O(1)（字母表固定 26）。
"""


def character_replacement(s, k):
    count = {}
    left = 0
    best = 0
    max_freq = 0
    for right, ch in enumerate(s):
        count[ch] = count.get(ch, 0) + 1
        max_freq = max(max_freq, count[ch])
        if (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


if __name__ == "__main__":
    assert character_replacement("AABABBA", 1) == 4
    assert character_replacement("ABAB", 2) == 4
    assert character_replacement("ABAB", 0) == 1
    assert character_replacement("A", 0) == 1
    assert character_replacement("AAAA", 2) == 4
    assert character_replacement("BAAAB", 2) == 5
    print("longest_repeating_character_replacement: all tests passed")
