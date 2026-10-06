"""395. 至少有 K 个重复字符的最长子串（Longest Substring with At Least K Repeating Characters）

题目：给你一个字符串 s 和一个整数 k，找出 s 中的最长子串，要求该子串中的
      每一个字符出现次数都不少于 k。返回该子串的长度。

思路（分治：用「出现次数不足 k 的字符」把串切开）：
    如果某个字符在整个串里出现的次数都不足 k，那么任何满足条件的子串都
    **不可能包含它**——因为子串里的次数只会更少。于是这个字符天然就是一道
    「分隔符」：合法的子串只可能落在它切出的某一段里。
    把所有这类「非法字符」都当作分隔符，把串切成若干段，递归地在每段里找
    最长合法子串，取最大即可。如果一段里所有字符的出现次数都不少于 k，
    整段就是合法的，直接返回它的长度。

    为什么可以放心把非法字符扔掉：合法子串对「每个含有的字符」都有下界要求，
    含了非法字符就永远满足不了；所以最优解一定不含任何非法字符，去掉它们
    不会丢解，反而缩小了问题规模。

    为什么分段递归是正确的：每一段内部的字符集合是原串的子集，用原串统计的
    「非法字符」在段内依然非法（次数只会更少或不变），所以按同样的规则继续
    切分是自洽的；递归到某段没有非法字符时，就是该段的答案。

    这类「按分隔符切分、各自递归」的分治，和 241 的「按运算符切分」是同一个
    味道：找到问题中「不可能被最优解跨过」的点，用它在中间一刀切开。

复杂度：最坏 O(n^2)（每层切分扫描 O(n)，最坏递归 O(n) 层，例如 k 很大时）；
    空间 O(n)（递归栈 + 子串切片）。n 为串长。
"""


def longest_substring(s, k):
    if len(s) < k:
        return 0
    for ch in set(s):
        if s.count(ch) < k:
            return max(longest_substring(part, k) for part in s.split(ch))
    return len(s)


if __name__ == "__main__":
    assert longest_substring("aaabb", 3) == 3
    assert longest_substring("ababbc", 2) == 5
    assert longest_substring("aaabbb", 3) == 6
    assert longest_substring("abc", 2) == 0
    assert longest_substring("a", 1) == 1
    assert longest_substring("ababacb", 3) == 0
    print("longest_substring: all tests passed")
