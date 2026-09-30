"""242. 有效的字母异位词（Valid Anagram）

题目：给定两个字符串 s 和 t，判断 t 是否是 s 的字母异位词
（每个字符出现的次数都相同，只是排列顺序不同）。

思路：异位词的本质是「字符计数完全一致」，所以两种等价做法：
    1. 计数哈希表：先统计 s 的字符频次，再遍历 t 逐一抵消，
       一旦某个字符频次不够（或最后还有剩余）就不是异位词；
    2. 排序后比较：排序后两串若相等即是异位词，代价 O(n log n)。
    这里用计数法，线性时间。因为题目字符集固定为小写字母，也可以用长度 26 的数组，
    不过哈希表对任意字符集都成立。

复杂度：时间 O(n)，空间 O(1)（小写字母只有 26 种，哈希表规模有上界）。
"""


def is_anagram(s, t):
    if len(s) != len(t):
        return False
    count = {}
    for ch in s:
        count[ch] = count.get(ch, 0) + 1
    for ch in t:
        if count.get(ch, 0) == 0:
            return False
        count[ch] -= 1
    return True


if __name__ == "__main__":
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False
    assert is_anagram("", "") is True
    assert is_anagram("aacc", "ccac") is False
    print("valid_anagram: all tests passed")
