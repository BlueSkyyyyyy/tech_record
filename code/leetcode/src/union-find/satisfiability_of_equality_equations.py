"""990. 等式方程的可满足性（Satisfiability of Equality Equations）

题目：给定一组形如 "a==b" 或 "a!=b" 的方程（变量只有小写字母），判断能否给所有变量
赋值，使全部方程同时成立。

思路（等价关系合并，再查矛盾）：
    把「==」看成「在同一集合」的关系，它具有自反、对称、传递性，正好是并查集做的事。
    分两趟：
    1. 第一趟只处理所有 "=="：把相等变量的并查集合并；相等具有传递性，
       所以 a==b、b==c 会把 a、b、c 合成一个集合；
    2. 第二趟处理所有 "!="：若某个不等式两端竟然落在同一集合里，说明前面已推出二者
       相等，矛盾，返回 False；全部检查通过则返回 True。

    为什么先处理等式再处理不等式：等式会不断「扩大」连通块，只有等所有等式都合并完，
    集合关系才最终确定，此时再验证不等式才准确；若一边合并一边检查不等式，
    可能因为合并顺序漏判。

复杂度：时间 O(n·α(26))，空间 O(26)。
"""
from dsu import DSU


def equations_possible(equations):
    dsu = DSU(26)
    for eq in equations:
        if eq[1] == '=':
            dsu.union(ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a'))
    for eq in equations:
        if eq[1] == '!' and dsu.find(ord(eq[0]) - ord('a')) == dsu.find(ord(eq[3]) - ord('a')):
            return False
    return True


if __name__ == "__main__":
    assert equations_possible(["a==b", "b!=a"]) is False
    assert equations_possible(["b==a", "a==b"]) is True
    assert equations_possible(["a==b", "b==c", "a==c"]) is True
    assert equations_possible(["a==b", "b!=c", "c==a"]) is False
    assert equations_possible(["c==c", "b==d", "x!=z"]) is True
    print("satisfiability_of_equality_equations: all tests passed")
