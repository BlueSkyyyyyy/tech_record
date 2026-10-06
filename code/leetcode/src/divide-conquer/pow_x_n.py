"""50. Pow(x, n)（快速幂）

题目：实现 pow(x, n)，计算 x 的 n 次幂（n 为整数，可能为负）。

思路（分治：指数每次减半）：
    朴素的连乘要做 n 次乘法，n 很大时慢。注意到：
        x^n = (x^(n/2))^2                （n 为偶数）
        x^n = (x^(n/2))^2 * x            （n 为奇数）
    也就是说，只要算出「一半指数」的幂，再平方一次（奇数再补乘一个 x），
    就能得到整个幂。指数每递归一层就减半，所以只需要 O(log n) 次乘法。

    为什么负指数能一并处理：x^(-n) = (1/x)^n，所以先把底数取倒数、指数取正，
    之后走同一套逻辑。

    为什么叫「快速幂」：它把「乘 n 次」压成「乘 log n 次」，是分治在数值计算里
    最经典的应用；同样的思路还能推广到矩阵快速幂、模意义下的快速幂。

复杂度：时间 O(log n)（指数每层减半），空间 O(log n)（递归栈）。
"""


def my_pow(x, n):
    if n < 0:
        x = 1.0 / x
        n = -n

    def power(base, exp):
        if exp == 0:
            return 1.0
        half = power(base, exp // 2)
        if exp % 2 == 0:
            return half * half
        return half * half * base

    return power(x, n)


if __name__ == "__main__":
    assert my_pow(2.0, 10) == 1024.0
    assert my_pow(2.0, 0) == 1.0
    assert my_pow(2.0, -2) == 0.25
    assert my_pow(0.5, -2) == 4.0
    assert my_pow(-2.0, 3) == -8.0
    assert my_pow(1.0, -2147483648) == 1.0
    print("pow_x_n: all tests passed")
