"""13. 罗马数字转整数（Roman to Integer）

题目：给定一个罗马数字字符串 s，将其转换成整数。罗马数字的七个基本字符为
I=1、V=5、X=10、L=50、C=100、D=500、M=1000；通常按「从左到右、大数在左小数在右
相加」的规则书写；当小数出现在大数左边时表示减去该小数（如 IV=4、IX=9）。

思路（从左往右扫，看右边一位决定加减）：
    从左到右遍历每个字符，取出它的值 v。如果它右边还有一个字符、且 v 比右边的值小，
    说明当前字符是「减法组合」的左半部分，应当减去 v；否则加上 v。累加结果即为答案。

    为什么这样就等价于罗马数字的规则：每个「小数在大数左边」的减法组合（IV、IX、
    XL、XC、CD、CM）都恰好由相邻两个字符构成，且左字符的值小于右字符。把这些少见
    的减法按「左字符取负」处理，其余正常相加，正好复现定义。无需特判具体是哪种组合，
    只要比较相邻两位大小即可。

    为什么不用先整体判断：逐位比较相邻两个字符是 O(n) 一遍扫描，通用且不含魔法；如果
    用「先查表替换子串」的写法，反而要小心替换顺序（比如先换 IX 再换 I）。

复杂度：时间 O(n)，空间 O(1)（查表是常量大小）。
"""


def roman_to_integer(s):
    values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    n = len(s)
    for i, ch in enumerate(s):
        v = values[ch]
        if i + 1 < n and v < values[s[i + 1]]:
            total -= v
        else:
            total += v
    return total


if __name__ == "__main__":
    assert roman_to_integer("III") == 3
    assert roman_to_integer("IV") == 4
    assert roman_to_integer("IX") == 9
    assert roman_to_integer("LVIII") == 58
    assert roman_to_integer("MCMXCIV") == 1994
    assert roman_to_integer("XL") == 40
    assert roman_to_integer("CDXLIV") == 444
    assert roman_to_integer("MMMCMXCIX") == 3999
    print("roman_to_integer: all tests passed")
