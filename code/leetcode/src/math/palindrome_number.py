"""9. 回文数（Palindrome Number）

题目：给定一个整数 x，如果 x 是一个回文数则返回 true，否则返回 false。回文数指
正序（从左向右）和倒序（从右向左）读都一样的整数。

思路（只反转一半数字）：
    最直观的做法是把整个数反转再比较是否相等，但只需反转一半就够了，还能顺手避免
    溢出：
        if x < 0 or (x % 10 == 0 and x != 0): return False
        rev = 0
        while x > rev:
            rev = rev * 10 + x % 10
            x //= 10
        return x == rev or x == rev // 10

    循环在「剩余部分 x 不再大于已反转部分 rev」时停下，此时反转已经吃掉一半位数。

    为什么可以只反转一半：回文在中间对称，越早发现两端对不上越早出结果。把数字从末位
    往高位搬进 rev、同时 x 自己不断右移，当 rev 追上或超过 x 时，二者位数已相当，各自
    就是原数的后半段与前半段。

    为什么要单独判「末尾是 0 且不为 0」：这类数（10、100、……）最高位是 1、最低位是 0，
    左右必然不同，但反转后位数会变短，只靠比较可能误判，所以提前返回。
    为什么最后要比较 rev // 10：当位数是奇数时，rev 会比 x 多出一位，那一位正是中间的
    数字，不影响回文，去掉它再比即可。

复杂度：时间 O(log x)（只处理一半位数），空间 O(1)。
"""


def palindrome_number(x):
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    rev = 0
    while x > rev:
        rev = rev * 10 + x % 10
        x //= 10
    return x == rev or x == rev // 10


if __name__ == "__main__":
    assert palindrome_number(121) is True
    assert palindrome_number(-121) is False
    assert palindrome_number(10) is False
    assert palindrome_number(0) is True
    assert palindrome_number(7) is True
    assert palindrome_number(12321) is True
    assert palindrome_number(123321) is True
    assert palindrome_number(12345) is False
    assert palindrome_number(1000021) is False
    print("palindrome_number: all tests passed")
