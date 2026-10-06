"""233. 数字 1 的个数（Number of Digit One）

题目：给定整数 n，统计所有小于等于 n 的非负整数中数字 1 出现的总次数。

思路（按「位」分别统计，不重不漏）：
    数 1 一共出现多少次 = 每个十进制位上出现 1 的次数之和。固定一个位
    （个位、十位、……权重 p = 10^k），把 0..n 的数按这一位切成三部分：

        high | cur | low      （high = n // (p*10)，cur = 该位的数字，low = n % p）

    - cur > 1：这一位取 1 时，high 可取 0..high 共 high+1 种，低位任意 0..p-1，
      贡献 (high + 1) * p；
    - cur == 1：高位取 0..high-1 时低位任取（high * p 种），高位取 high 时低位
      只能取 0..low（low + 1 种），贡献 high * p + low + 1；
    - cur < 1（即 0）：这一位取 1 时高位只能取 0..high-1，贡献 high * p。

    逐位相加即可，无需枚举任何数字。

复杂度：时间 O(log n)（位数），空间 O(1)。
"""


def count_digit_one(n):
    count = 0
    p = 1
    while p <= n:
        high = n // (p * 10)
        cur = (n // p) % 10
        low = n % p
        if cur > 1:
            count += (high + 1) * p
        elif cur == 1:
            count += high * p + low + 1
        else:
            count += high * p
        p *= 10
    return count


if __name__ == "__main__":
    assert count_digit_one(0) == 0
    assert count_digit_one(13) == 6
    assert count_digit_one(1) == 1
    assert count_digit_one(10) == 2
    assert count_digit_one(99) == 20
    assert count_digit_one(100) == 21
    assert count_digit_one(824883294) == 767944060
    print("count_digit_one: all tests passed")
