"""383. 赎金信（Ransom Note）

题目：给定两个字符串 ransomNote 和 magazine，判断 ransomNote 能否由 magazine 中的
字符拼成。magazine 中的每个字符只能在 ransomNote 里使用一次。
例如 ransomNote="aa", magazine="aab" 可以，magazine="ab" 不行。

思路：「能不能拼成」等价于「ransomNote 里每个字符的出现次数都不超过 magazine」。
     先统计 magazine 的字符频次，再遍历 ransomNote 逐个扣减：
       - 若某个字符计数已为 0，说明 magazine 供不上，返回 False；
       - 否则计数减一。
     全部扣完仍没出问题，就能拼成。
     这正是「可不可行 = 资源约束」的判定，与 242 有效的字母异位词是同一套抵消逻辑，
     区别是本题只要求「够用」（子集关系），不要求两个字符串完全等量。

复杂度：时间 O(m + n)，空间 O(字符集大小)。
"""


def can_construct(ransom_note, magazine):
    count = {}
    for ch in magazine:
        count[ch] = count.get(ch, 0) + 1
    for ch in ransom_note:
        if count.get(ch, 0) == 0:
            return False
        count[ch] -= 1
    return True


if __name__ == "__main__":
    assert can_construct("a", "b") is False
    assert can_construct("aa", "ab") is False
    assert can_construct("aa", "aab") is True
    assert can_construct("", "anything") is True
    assert can_construct("abc", "cba") is True
    print("ransom_note: all tests passed")
