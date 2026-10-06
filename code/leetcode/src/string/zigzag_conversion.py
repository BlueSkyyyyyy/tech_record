"""6. Z 字形变换（Zigzag Conversion）

题目：把字符串 s 按从上到下、再从左下到右上「之」字形填进 numRows 行，最后逐行读出，
返回读出的字符串。

思路（按行分桶模拟）：
    与其在二维网格里画 Z 字，不如反过来：给每一行准备一个「桶」，让字符按 Z 字的行走
    顺序依次落进对应的桶里，最后把各桶从上到下拼起来即可。

    行走规律只有一条：当前行 `cur` 在 0 和 numRows-1 之间来回移动——走到最上面就改向下
    （step = +1），走到最下面就改向上（step = -1）。于是每读一个字符，先放进 `rows[cur]`，
    再更新方向、移动 `cur`。

    为什么不需要真的建二维表：Z 字路径本身只依赖 `cur` 和方向，两个变量就能完整描述；
    而某一行里字符的左右顺序，恰好就是它们被访问的先后顺序，所以顺序追加即可。

    边界：numRows == 1 时方向永远不变，直接返回原串即可；numRows >= len(s) 时每个字符
    各占一行、读出来还是原串，一并提前返回。

复杂度：时间 O(n)（每个字符访问一次），空间 O(n)（各行的字符总量）。
"""


def convert(s, numRows):
    if numRows == 1 or numRows >= len(s):
        return s

    rows = [""] * numRows
    cur = 0
    step = 1
    for ch in s:
        rows[cur] += ch
        if cur == 0:
            step = 1
        elif cur == numRows - 1:
            step = -1
        cur += step
    return "".join(rows)


if __name__ == "__main__":
    assert convert("PAYPALISHIRING", 3) == "PAHNAPLSIIGYIR"
    assert convert("PAYPALISHIRING", 4) == "PINALSIGYAHRPI"
    assert convert("A", 1) == "A"
    assert convert("AB", 1) == "AB"
    assert convert("ABC", 5) == "ABC"
    assert convert("HELLO", 2) == "HLOEL"
    print("zigzag_conversion: all tests passed")
