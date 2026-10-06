"""12. 整数转罗马数字（Integer to Roman）

题目：给定一个整数 num（1 <= num <= 3999），将它转换成罗马数字。

思路（贪心：从大到小尽量取）：
    把「面值」按从大到小排好，包含所有基本字符以及六个减法组合：
        1000 M, 900 CM, 500 D, 400 CD, 100 C, 90 XC,
        50 L, 40 XL, 10 X, 9 IX, 5 V, 4 IV, 1 I
    从头到尾扫描这张表，只要 num 还够减，就重复追加对应的符号、并从 num 里减去该面值。

    为什么贪心是对的：罗马数字的标准写法要求从左到右面值非增，且每个位置上的符号一旦
    确定就不应再用更小的符号去「补」。把 900、400、90、40、9、4 这些减法组合也当成
    独立面值，就保证了每一步取到的都是当前能用的最大面值，取完一个再取下一个更大的
    位，最终拼出的正是唯一的标准表示。整数范围内的标准罗马数字表示是唯一的，所以贪心
    不会出现「取了大的反而凑不出」的情况。

    为什么不能只用七个基本字符：那样会出现 IIII 表示 4 之类的非标准写法，也不符合题目
    要求的规范形式，因此必须把减法组合一并列进面值表。

复杂度：时间 O(1)（面值表固定 13 项，循环次数只与 num 的位数有关），空间 O(1)
（不计输出字符串）。
"""


def integer_to_roman(num):
    pairs = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    res = []
    for value, sym in pairs:
        while num >= value:
            res.append(sym)
            num -= value
    return "".join(res)


if __name__ == "__main__":
    assert integer_to_roman(3) == "III"
    assert integer_to_roman(4) == "IV"
    assert integer_to_roman(9) == "IX"
    assert integer_to_roman(58) == "LVIII"
    assert integer_to_roman(1994) == "MCMXCIV"
    assert integer_to_roman(40) == "XL"
    assert integer_to_roman(444) == "CDXLIV"
    assert integer_to_roman(3999) == "MMMCMXCIX"
    print("integer_to_roman: all tests passed")
