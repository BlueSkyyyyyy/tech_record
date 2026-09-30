"""125. 验证回文串（Valid Palindrome）

题目：给定字符串 s，只考虑其中的字母和数字，并忽略大小写，判断它是否为回文串。

思路：「对撞双指针」。
    lo 指向头部，hi 指向尾部，向中间收缩。每次先把 lo、hi 挪到下一个
    字母/数字上（跳过标点和空格），再比较它们的小写形式。
    为什么可以这样做：回文只关心正读与反读是否相同，而判定这对字符
    是否相等只依赖它们本身。跳过非字母数字字符不会影响回文性质，
    因此一次线性扫描即可。

复杂度：时间 O(n)，空间 O(1)。
"""


def is_palindrome(s):
    lo, hi = 0, len(s) - 1
    while lo < hi:
        while lo < hi and not s[lo].isalnum():
            lo += 1
        while lo < hi and not s[hi].isalnum():
            hi -= 1
        if s[lo].lower() != s[hi].lower():
            return False
        lo += 1
        hi -= 1
    return True


if __name__ == "__main__":
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("race a car") is False
    assert is_palindrome(" ") is True
    assert is_palindrome("") is True
    assert is_palindrome("0P") is False
    print("valid_palindrome: all tests passed")
