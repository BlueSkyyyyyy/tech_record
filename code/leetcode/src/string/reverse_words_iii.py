"""557. 反转字符串中的单词 III（Reverse Words in a String III）

题目：给定字符串 s（单词之间用单个空格分隔），反转每个单词内部的字符顺序，同时保持单词
之间的顺序和空格位置不变。

思路（按空格分段 + 段内反转）：
    与 151 不同，这里的单词顺序**不变**，所以不需要「整体反转」这一步，只需找出每个单词
    的区间再各自反转即可。用一个 start 记录当前单词起点，从左往右扫；每遇到一个空格（或
    扫到串尾），就把区间 [start, i-1] 交给 344 的对撞双指针反转，然后把 start 移到 i+1。

    为什么用「扫描到 n」作为终止条件之一：最后一个单词后面没有空格，只靠「遇到空格」会漏
    掉它。让 i 走到 n 时也触发一次结算，就能统一处理末尾单词，而不用额外补一段收尾代码。

    这也是「按分隔符切分块、块内独立处理」的通用骨架：只要块内操作是局部的、块间互不影响，
    就可以边扫描边结算，无需真正把字符串 split 成数组。

复杂度：时间 O(n)，空间 O(n)（Python 需要转成字符列表）/ O(1) 额外空间（C++ 原地）。
"""


def reverse_words_iii(s):
    chars = list(s)
    n = len(chars)

    def reverse(lo, hi):
        while lo < hi:
            chars[lo], chars[hi] = chars[hi], chars[lo]
            lo += 1
            hi -= 1

    start = 0
    for i in range(n + 1):
        if i == n or chars[i] == " ":
            reverse(start, i - 1)
            start = i + 1
    return "".join(chars)


if __name__ == "__main__":
    assert reverse_words_iii("Let's take LeetCode contest") == "s'teL ekat edoCteeL tsetnoc"
    assert reverse_words_iii("Mr Ding") == "rM gniD"
    assert reverse_words_iii("a") == "a"
    assert reverse_words_iii("") == ""

    print("reverse_words_iii: all tests passed")
