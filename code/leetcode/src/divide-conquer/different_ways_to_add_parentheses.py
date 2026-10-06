"""241. 为运算表达式设计优先级（Different Ways to Add Parentheses）

题目：给你一个由数字和运算符（+、-、*）组成的字符串 expression，按不同的
      加括号方式，返回所有可能的运算结果（顺序不限，允许重复）。

思路（分治：以每个运算符为「最外层」切一刀）：
    任何一种加括号方式，最终都会归结为「某一次运算作为整式的最后一步」。
    所以可以枚举表达式里的每一个运算符，把它当作最外层运算：
      以它为界，左边是一个子表达式、右边是一个子表达式，分别递归求出它们
      所有可能的结果，再做一次这个运算符的组合，把结果收集起来。
    当表达式里没有运算符（就是一个数字）时，它就是最小子问题，直接返回这个数。

    为什么这样能不重不漏地枚举所有加括号方式：每种加括号方式都有唯一的
    「最后执行的运算符」，按这个运算符分类，正好一一对应；递归到子表达式时
    同样枚举它自己的最后一步。这本质上是在枚举「不同形态的表达式树」。

    为什么会出现重复结果：不同的括号方式可能算出相同的值（例如 2*3-4*5 的
    两种方式都得到 -10），题目允许结果重复，所以不用去重。

    注意：这里数字都是非负整数，运算符只有 +、-、*，不涉及整数除法与符号。

复杂度：时间与「卡特兰数」同阶（第 n 个运算符有 Catalan(n) 种括号方式，
    递归会重复计算子表达式，最坏指数级；加记忆化可降为多项式）。
    空间 O(n)（递归栈）。n 为运算符个数，规模很小，直接分治足够。
"""


def diff_ways_to_compute(expression):
    if expression.isdigit():
        return [int(expression)]

    results = []
    for i, ch in enumerate(expression):
        if ch in "+-*":
            left = diff_ways_to_compute(expression[:i])
            right = diff_ways_to_compute(expression[i + 1:])
            for a in left:
                for b in right:
                    if ch == "+":
                        results.append(a + b)
                    elif ch == "-":
                        results.append(a - b)
                    else:
                        results.append(a * b)
    return results


if __name__ == "__main__":
    assert sorted(diff_ways_to_compute("2-1-1")) == [0, 2]
    assert sorted(diff_ways_to_compute("2*3-4*5")) == [-34, -14, -10, -10, 10]
    assert sorted(diff_ways_to_compute("3")) == [3]
    assert sorted(diff_ways_to_compute("1+2+3")) == [6, 6]
    print("diff_ways_to_compute: all tests passed")
