"""7. 整数反转（Reverse Integer）

题目：给定一个 32 位有符号整数 x，返回将 x 中的数字部分反转后的结果；如果反转后
整数超过 32 位有符号整数的范围 [-2^31, 2^31 - 1]，就返回 0。假设环境不允许存储
64 位整数。

思路（逐位弹出 + 推入）：
    反复取出 x 的末位数字，把它「推入」到结果 rev 的末尾：
        digit = x % 10
        rev = rev * 10 + digit
        x //= 10
    所谓「推入」，就是先把已有结果整体左移一位（十进制下乘 10），再把新数字放到个位。

    为什么这样能反转：设原数是 d_k d_{k-1} ... d_0，每次取末位相当于从右往左读一位。
    rev 每轮乘 10 再放新位，新读到的位就落在越靠左的位置，于是读取顺序虽然是从右往左，
    写进 rev 的顺序却使它们最终左高右低地排好，正好等于原数倒过来。

    溢出处理：Python 的整数不会溢出，所以在最后统一判断是否越界即可。注意题目要的是
    反转后的范围判断，而不是反转前的。另外负数取模在 Python 中结果非负（如 -123 % 10 = 7），
    所以先把符号拆出来、只反转绝对值，最后再补回符号，逻辑最干净。

复杂度：时间 O(log|x|)（十进制位数），空间 O(1)。
"""


def reverse_integer(x):
    sign = -1 if x < 0 else 1
    x = abs(x)
    rev = 0
    while x:
        rev = rev * 10 + x % 10
        x //= 10
    rev *= sign
    if rev < -(2 ** 31) or rev > 2 ** 31 - 1:
        return 0
    return rev


if __name__ == "__main__":
    assert reverse_integer(123) == 321
    assert reverse_integer(-123) == -321
    assert reverse_integer(120) == 21
    assert reverse_integer(0) == 0
    assert reverse_integer(7) == 7
    assert reverse_integer(1534236469) == 0
    assert reverse_integer(-2147483648) == 0
    assert reverse_integer(2147483647) == 0
    print("reverse_integer: all tests passed")
