"""784. 字母大小写全排列（Letter Case Permutation）

题目：给定字符串 s，其中的字母可以任意改成大写或小写，数字保持不变，返回所有能
得到的字符串。

思路（只对"需要决策的字母"开开关）：
    每个字母有两种选择（原样/切换大小写），所以答案共有 2^(字母个数) 个。用一个
    mask 的第 j 位表示"第 j 个字母要不要切换大小写"：0 用小写、1 用大写。数字完全不
    参与——先把字母的下标收集成 letters 列表，枚举 mask 时只动这些位置。

    和第 78 题是同一个套路：真正的"开关"只有字母，"集合"就是"哪些位置切成了大写"。
    先把可选对象挑出来（letters），再枚举子集，是这个模板的标准起手式。

复杂度：时间 O(2^L * n)（L 是字母数，每个结果要复制并改写整串），空间 O(2^L * n)。
"""


def letter_case_permutation(s):
    letters = [i for i, ch in enumerate(s) if ch.isalpha()]
    res = []
    for mask in range(1 << len(letters)):
        chars = list(s)
        for j, i in enumerate(letters):
            chars[i] = chars[i].upper() if mask >> j & 1 else chars[i].lower()
        res.append("".join(chars))
    return res


if __name__ == "__main__":
    assert sorted(letter_case_permutation("a1b2")) == ["A1B2", "A1b2", "a1B2", "a1b2"]
    assert sorted(letter_case_permutation("3z4")) == ["3Z4", "3z4"]
    assert letter_case_permutation("12345") == ["12345"]
    assert sorted(letter_case_permutation("ab")) == ["AB", "Ab", "aB", "ab"]
    print("letter_case_permutation: all tests passed")
