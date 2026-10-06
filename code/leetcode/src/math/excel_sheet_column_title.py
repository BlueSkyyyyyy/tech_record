"""168. Excel 表列名称（Excel Sheet Column Title）

题目：给定一个正整数 columnNumber，返回它在 Excel 表中对应的列名称。规则与 171
互逆：1 -> "A"、26 -> "Z"、27 -> "AA"、28 -> "AB"。

思路（每次先减 1 再取模 26）：
    当 n 大于 0 时反复：
        n -= 1
        res.append(chr(ord('A') + n % 26))
        n //= 26
    最后把收集到的字符倒序拼接。

    为什么必须「先减 1」：这是一个「1 起始」的 26 进制——A 对应 1，但取模运算
    `n % 26` 的取值范围是 0~25。若不减 1，26 会先对 26 取模得 0（映射成 'A'），
    再整除得 1，反而又贡献一个 'A'，得到 "AA"，但正确答案是 "Z"。把 n 先减 1，等于
    把「1~26」平移成「0~25」，让 26 落到 25 上、正确映射成 'Z'。

    为什么从低位往高位取、最后要倒序：和所有进制转换一样，% 26 先取出的是最低位
    （最右边的字母），所以要先攒起来，最后整体反转。

复杂度：时间 O(log n)，空间 O(log n)（输出字符串长度）。
"""


def excel_sheet_column_title(column_number):
    res = []
    n = column_number
    while n > 0:
        n -= 1
        res.append(chr(ord("A") + n % 26))
        n //= 26
    return "".join(reversed(res))


if __name__ == "__main__":
    assert excel_sheet_column_title(1) == "A"
    assert excel_sheet_column_title(2) == "B"
    assert excel_sheet_column_title(26) == "Z"
    assert excel_sheet_column_title(27) == "AA"
    assert excel_sheet_column_title(28) == "AB"
    assert excel_sheet_column_title(701) == "ZY"
    assert excel_sheet_column_title(52) == "AZ"
    assert excel_sheet_column_title(18278) == "ZZZ"
    print("excel_sheet_column_title: all tests passed")
