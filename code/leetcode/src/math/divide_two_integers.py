"""29. 两数相除（Divide Two Integers）

题目：给定两个整数 dividend 和 divisor，在不使用乘法、除法和取模运算的前提下，
计算两数相除的商（向零截断）。结果应在 32 位有符号整数范围内，若溢出则返回
2^31 - 1。

思路（位运算倍增：把除法拆成 2 的幂之和）：
    任何数都能用二进制表示，商也可以写成若干个 2 的幂之和。于是我们反复从被除数里减掉
    「除数 × 2^k」这种最大的、仍然装得下的部分，同时把 2^k 累加进商：
        while a >= b:
            shift = 0
            while a >= (b << (shift + 1)):
                shift += 1
            a -= b << shift
            result += 1 << shift

    为什么这样能得到正确的商且不超时：如果每次只减一个 divisor，最坏要减 2^31 次。
    倍增把「需要多少个 divisor」按二进制位成组地扣除，每一轮至少减掉当前最大的一堆，
    轮数降到 O(log^2 n)。等价于在做「用二进制逐位试商」。

    为什么要单独处理两种边界：符号用「两数异号」判断；INT_MIN 取绝对值会溢出，所以
    Python 里用无界整数天然安全，C++ 里则先转 long long 再取绝对值。唯一的溢出结果
    是 INT_MIN / -1（等于 2^31，超出上界），需要提前挡掉。

    为什么结果向零截断：题目要求截断而非取整，而我们对绝对值做除法、最后补符号，恰好
    实现向零截断。

复杂度：时间 O(log^2 n)，空间 O(1)。
"""


def divide_two_integers(dividend, divisor):
    if dividend == -(2 ** 31) and divisor == -1:
        return 2 ** 31 - 1
    negative = (dividend < 0) != (divisor < 0)
    a = abs(dividend)
    b = abs(divisor)
    result = 0
    while a >= b:
        shift = 0
        while a >= (b << (shift + 1)):
            shift += 1
        a -= b << shift
        result += 1 << shift
    return -result if negative else result


if __name__ == "__main__":
    assert divide_two_integers(10, 3) == 3
    assert divide_two_integers(7, -3) == -2
    assert divide_two_integers(-7, 3) == -2
    assert divide_two_integers(-7, -3) == 2
    assert divide_two_integers(0, 1) == 0
    assert divide_two_integers(1, 1) == 1
    assert divide_two_integers(-1, -1) == 1
    assert divide_two_integers(1, 2) == 0
    assert divide_two_integers(-(2 ** 31), -1) == 2 ** 31 - 1
    assert divide_two_integers(-(2 ** 31), 1) == -(2 ** 31)
    print("divide_two_integers: all tests passed")
