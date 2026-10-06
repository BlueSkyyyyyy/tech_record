"""8. 字符串转换整数 atoi（String to Integer）

题目：把字符串按下面的规则转成整数：先跳过前导空格；可选读一个 `+` 或 `-` 号；再读连续
数字；遇到第一个非数字字符就停止，后面的内容忽略。若读不到数字返回 0。结果要夹在 32 位
有符号整数区间 [−2^31, 2^31−1] 内，越界则返回对应的边界值。

思路（先剥前缀，再逐位累乘，越界即夹紧）：
    三步流水线，正好对应题面的三段描述：

    1. **跳过前导空格**：`while s[i] == ' '` 一直走。
    2. **读符号**：若当前是 `+` 或 `-`，记下 `sign` 后右移一位。没有符号默认正。
    3. **逐位累加**：只要当前是数字，就 `num = num * 10 + (s[i] - '0')`，并右移一位。

    关键在越界处理：题目要求「超出范围就返回边界值」。做法是**每累加一位就检查一次**，
    一旦 `sign * num` 出了 32 位区间，立即返回 `INT_MAX` 或 `INT_MIN`。这样既不用真正
   存下一个超过 32 位的数，也天然符合「一旦越界就定格在边界」的语义。

    注意 `+0`、`000123` 这类前导零不必特殊处理：逐位累乘会自然得到 0 或 123。

复杂度：时间 O(n)（每个字符最多看一遍），空间 O(1)。
"""


def my_atoi(s):
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31

    n = len(s)
    i = 0
    while i < n and s[i] == " ":
        i += 1

    sign = 1
    if i < n and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1
        i += 1

    num = 0
    while i < n and s[i].isdigit():
        num = num * 10 + (ord(s[i]) - ord("0"))
        if sign * num > INT_MAX:
            return INT_MAX
        if sign * num < INT_MIN:
            return INT_MIN
        i += 1

    return sign * num


if __name__ == "__main__":
    assert my_atoi("42") == 42
    assert my_atoi("   -42") == -42
    assert my_atoi("4193 with words") == 4193
    assert my_atoi("words and 987") == 0
    assert my_atoi("-91283472332") == -2**31
    assert my_atoi("2147483648") == 2**31 - 1
    assert my_atoi("+1") == 1
    assert my_atoi("   +0 123") == 0
    assert my_atoi("0000123") == 123
    assert my_atoi("") == 0
    assert my_atoi("+-12") == 0
    print("string_to_integer_atoi: all tests passed")
