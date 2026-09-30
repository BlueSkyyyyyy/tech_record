"""20. 有效的括号（Valid Parentheses）

题目：给定一个只包括 '('、')'、'['、']'、'{'、'}' 的字符串 s，
      判断字符串是否有效。有效需同时满足：左括号必须用相同类型的右括号闭合，
      且必须以正确的顺序闭合。

思路：从左到右扫描。遇到左括号就压栈；遇到右括号，就检查栈顶是不是与它配对的左括号：
        是，则弹掉栈顶（这一对闭合了）；
        否（或栈为空），说明顺序错了，直接返回 False。
      扫完以后，栈必须为空才说明所有左括号都被闭合。

      为什么用栈：括号匹配的规则是「后出现的左括号必须先被闭合」，这正是栈「后进先出」
      的天然语义。每遇到一个右括号，要配对的永远是「最近一个还没闭合的左括号」，也就是栈顶。

      为什么扫描结束还要判空栈：像 "(((" 这种只有左括号的串，扫描过程不会出错，
      但存在未闭合的括号，所以必须靠「最后栈为空」来兜住。

复杂度：时间 O(n)（每个字符进出栈各一次），空间 O(n)（最坏全是左括号）。
"""


def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack


if __name__ == "__main__":
    assert is_valid("()") is True
    assert is_valid("()[]{}") is True
    assert is_valid("(]") is False
    assert is_valid("([)]") is False
    assert is_valid("{[]}") is True
    assert is_valid("(") is False
    assert is_valid(")") is False
    assert is_valid("") is True
    print("valid_parentheses: all tests passed")
