"""172. 阶乘后的零（Factorial Trailing Zeroes）

题目：给定整数 n，返回 n! 结果末尾有多少个零。

思路（数因子 5 的个数）：
    阶乘末尾的每个 0 都来自一对因子 (2, 5)（因为 10 = 2 * 5）。在 n! 里，因子 2 的
    数量远多于因子 5，所以零的个数由因子 5 的个数决定。把 5 的贡献分层累加：
        count = 0
        while n > 0:
            n //= 5
            count += n
    第一次 n // 5 数出「至少含一个 5」的数的个数，第二次数出「至少含两个 5」（即 25 的
    倍数）的个数，依此类推，恰好把每个数里的 5 因子按层补齐。

    为什么不能直接算 n! 再数零：n 稍大阶乘就会溢出，而且完全没必要——零的个数只取决于
    5 的幂次，是一个纯粹的计数问题。

    为什么除以 5 的循环次数是 log 的量级：n 每次缩到五分之一，n <= 10^4 时几轮就归零，
    效率很高。

复杂度：时间 O(log_5 n)，空间 O(1)。
"""


def factorial_trailing_zeroes(n):
    count = 0
    while n > 0:
        n //= 5
        count += n
    return count


if __name__ == "__main__":
    assert factorial_trailing_zeroes(3) == 0
    assert factorial_trailing_zeroes(5) == 1
    assert factorial_trailing_zeroes(10) == 2
    assert factorial_trailing_zeroes(25) == 6
    assert factorial_trailing_zeroes(125) == 31
    assert factorial_trailing_zeroes(0) == 0
    assert factorial_trailing_zeroes(1) == 0
    print("factorial_trailing_zeroes: all tests passed")
