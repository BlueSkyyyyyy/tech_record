"""151. 翻转字符串里的单词（Reverse Words in a String）

题目：给定字符串 s，翻转其中单词的顺序。单词由非空格字符组成，单词之间可能有多个空格。
要求结果中单词之间只用一个空格隔开，且不含首尾空格。

思路（整体反转 + 每个单词再反转 + 就地压缩空格）：
    这是 344「反转」这个动作的组合运用，分三步：

    1. **整体反转**整个字符序列。此时单词的先后顺序被颠倒，但每个单词内部的字母也被颠倒
       了（"the sky" -> "yks eht"）。
    2. **逐个单词再反转一次**，把每个单词内部的字母顺序复原（"yks" -> "sky"），而单词之间
       的相对顺序保持颠倒后的结果。
    3. **就地压缩空格**：一边从左往右读单词，一边把需要的字符写到前面。读到一个新单词时，
       如果前面已经写过内容，就先补一个空格。写完后把这段写入区间反转（即第 2 步）。

    为什么可行：整体反转把「单词顺序」翻转了一次、也把「每个单词内部」翻转了一次；对每个
    单词再做一次反转，正好把内部的翻转抵消掉，于是只剩下单词顺序被翻转——正是题目要的。

    这种「两次反转抵消一次」的技巧还常用于数组循环移位（见数组与双指针篇的 189 轮转数组：
    先整体反转、再分段反转）。

复杂度：时间 O(n)（整体反转、逐词反转、压缩各扫一遍），空间 O(n)（Python 需把字符串
转成字符列表）或 O(1) 额外空间（C++ 直接在 std::string 上原地完成）。
"""


def reverse_words(s):
    chars = list(s)
    n = len(chars)

    def reverse(lo, hi):
        while lo < hi:
            chars[lo], chars[hi] = chars[hi], chars[lo]
            lo += 1
            hi -= 1

    reverse(0, n - 1)

    write = 0
    i = 0
    while i < n:
        if chars[i] != " ":
            if write > 0:
                chars[write] = " "
                write += 1
            start = write
            while i < n and chars[i] != " ":
                chars[write] = chars[i]
                write += 1
                i += 1
            reverse(start, write - 1)
        else:
            i += 1
    return "".join(chars[:write])


if __name__ == "__main__":
    assert reverse_words("the sky is blue") == "blue is sky the"
    assert reverse_words("  hello world  ") == "world hello"
    assert reverse_words("a good   example") == "example good a"
    assert reverse_words("") == ""
    assert reverse_words("   ") == ""
    assert reverse_words("word") == "word"
    assert reverse_words("  a  ") == "a"
    assert reverse_words("Epic   systems   rocks") == "rocks systems Epic"
    print("reverse_words_in_a_string: all tests passed")
