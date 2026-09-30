"""150. 逆波兰表达式求值（Evaluate Reverse Polish Notation）

题目：给定一个按逆波兰表示法（后缀表达式）排列的字符串数组 tokens，
      求该表达式的值。合法的运算符只有 +、-、*、/，除法向零截断。

思路（一个栈存操作数）：
    从左到右读 token。遇到数字就压栈；遇到运算符就弹出栈顶两个数，
    先弹出的是右操作数 b、后弹出的是左操作数 a，算完 a op b 再压回栈。
    扫完后栈里只剩一个数，就是答案。

    为什么用栈：后缀表达式把运算顺序「藏」在了 token 的先后里，
    每次运算的两个操作数，正是最近两个还没被消耗的数——最近优先，就是栈顶。
    先遇到的数先入栈、沉得深，后遇到的数在栈顶，所以做运算时先弹出的是右操作数。

    为什么除法要特别处理：LeetCode 规定除法向零截断（如 -7/2 = -3），
    Python 的 // 是向下取整（-7//2 = -4），所以这里用 int(a / b) 而不是 a // b。

复杂度：时间 O(n)（每个 token 处理一次），空间 O(n)（最坏全是数字）。
"""


def eval_rpn(tokens):
    stack = []
    ops = {"+", "-", "*", "/"}
    for tok in tokens:
        if tok in ops:
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(tok))
    return stack[-1]


if __name__ == "__main__":
    assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
    assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22
    assert eval_rpn(["-7", "2", "/"]) == -3
    assert eval_rpn(["3"]) == 3
    print("eval_rpn: all tests passed")
