"""171. Excel 表列序号（Excel Sheet Column Number）

题目：给定一个 Excel 表格的列名称（如 "A"、"AB"、"ZY"），返回其对应的列序号。
规则是 A -> 1、B -> 2、...、Z -> 26、AA -> 27、AB -> 28，本质像一个「没有 0」的
26 进制。

思路（按位累乘）：
    从左到右处理每个字符，把结果更新为：
        res = res * 26 + (当前字符的值)
    其中字符的值 = 字符在字母表中的序号，'A' -> 1、'B' -> 2、...、'Z' -> 26。

    为什么是「乘 26 再加」：这和一个十进制数从高位到低位逐位读出完全一样。比如十进制
    的 253 = ((2 * 10) + 5) * 10 + 3。这里只是把进制从 10 换成 26，而每一位的值从
    0~25 变成了 1~26（因为 A 表示 1 而不是 0）。逐位「进位展开」即可得到整数值。

    为什么不能用 `ord(ch) - ord('A')` 当值：那样 A 会算成 0，整个写法就退化成了普通
    的 0 起始 26 进制，AA 会被算成 0 * 26 + 0 = 0，与题意不符，所以必须再加 1。

复杂度：时间 O(n)（n 为字符串长度），空间 O(1)。
"""


def excel_sheet_column_number(s):
    res = 0
    for ch in s:
        res = res * 26 + (ord(ch) - ord("A") + 1)
    return res


if __name__ == "__main__":
    assert excel_sheet_column_number("A") == 1
    assert excel_sheet_column_number("B") == 2
    assert excel_sheet_column_number("Z") == 26
    assert excel_sheet_column_number("AA") == 27
    assert excel_sheet_column_number("AB") == 28
    assert excel_sheet_column_number("ZY") == 701
    assert excel_sheet_column_number("AZ") == 52
    assert excel_sheet_column_number("ZZZ") == 18278
    print("excel_sheet_column_number: all tests passed")
