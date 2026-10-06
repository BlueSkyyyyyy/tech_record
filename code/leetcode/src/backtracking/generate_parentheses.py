"""22. 括号生成（Generate Parentheses）

题目：数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的
并且有效的括号组合。

思路（带约束的回溯，剪掉非法前缀）：
    从左往右一个字符一个字符地决定放 '(' 还是 ')'。如果毫无限制地枚举，会有
    2^(2n) 种串，但其中绝大多数不合法。关键在于：括号串合法的充要条件是
    任意前缀中左括号数 ≥ 右括号数，且总左括号数 == 总右括号数。

    于是把这两条约束直接写进递归参数：
      - 还能放左括号当且仅当 open < n；
      - 还能放右括号当且仅当 close < open（保证任意前缀左不少于右）。
    只在满足条件时才递归，非法前缀根本不会被生成，等于在构建过程中就剪枝。

    为什么这样剪最有效：合法性是「前缀性质」，一旦某前缀非法，往后怎么补都
    救不回来。所以剪枝可以放在每一步做，而不是生成完整串再校验。

复杂度：时间 O(C(2n,n)·n)（结果数为第 n 个卡特兰数，构造每个串 O(n)），
    空间 O(n)（递归深度，不含答案本身）。
"""


def generate_parenthesis(n):
    res = []

    def backtrack(cur, open_, close):
        if len(cur) == 2 * n:
            res.append(cur)
            return
        if open_ < n:
            backtrack(cur + "(", open_ + 1, close)
        if close < open_:
            backtrack(cur + ")", open_, close + 1)

    backtrack("", 0, 0)
    return res


if __name__ == "__main__":
    out = generate_parenthesis(3)
    assert len(out) == 5
    assert "((()))" in out
    assert "()()()" in out
    assert "())(" not in out

    assert generate_parenthesis(1) == ["()"]
    assert generate_parenthesis(0) == [""]
    print("generate_parenthesis: all tests passed")
