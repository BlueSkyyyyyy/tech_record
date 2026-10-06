"""17. 电话号码的字母组合（Letter Combinations of a Phone Number）

题目：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
答案可以按任意顺序返回。数字到字母的映射与电话按键相同（2->abc, 3->def,
4->ghi, 5->jkl, 6->mno, 7->pqrs, 8->tuv, 9->wxyz）。

思路（逐位选择的回溯）：
    这是最简单的「多叉决策树」：每个数字对应一组候选字母，从左到右逐位决定
    这一位选哪个字母。树的深度就是数字个数，每层分叉数就是该数字的字母数。

    用下标 i 表示处理到第几个数字。选完一个字母压入 path，递归处理 i+1，
    回来再弹出。当 i 走到末尾时，把 path 拼成一个字符串收进答案。

    为什么用拼接字符串而不是数字下标去重：每一位对应一个独立按键，天然不会
    有「重复元素」问题，也没有顺序歧义，所以既不需要 start 也不需要 used，
    只要逐位枚举即可。这是回溯最朴素的形态。

复杂度：时间 O(4^n·n)（n 为数字个数，每位最多 4 个字母，拼接结果 O(n)），
    空间 O(n)（递归深度，不含答案本身）。
"""


def letter_combinations(digits):
    if not digits:
        return []
    table = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz",
    }
    res = []
    path = []

    def backtrack(i):
        if i == len(digits):
            res.append("".join(path))
            return
        for ch in table[digits[i]]:
            path.append(ch)
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res


if __name__ == "__main__":
    out = letter_combinations("23")
    assert len(out) == 9
    assert "ad" in out and "ae" in out and "cf" in out

    assert letter_combinations("") == []
    assert letter_combinations("2") == ["a", "b", "c"]
    assert letter_combinations("7") == ["p", "q", "r", "s"]
    print("letter_combinations: all tests passed")
