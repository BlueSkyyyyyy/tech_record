"""1067. 范围内的数字计数（Digit Count in Range）

题目：给定整数 d（0..9）、low、high，统计数字 d 在区间 [low, high] 的所有整数
中出现的总次数。

思路（前缀相减 + 按位统计）：
    先做一个辅助函数 count_upto(n, d)：统计 0..n 中 d 出现多少次，那么区间答案
    就是 count_upto(high, d) - count_upto(low-1, d)。这就是「前缀和」的思想，
    只不过前缀量是「某数字出现的次数」。

    count_upto 对每一位（权重 p）分三段：high | cur | low。

    d != 0 时：
      - cur > d：高位 0..high 都可，贡献 (high + 1) * p；
      - cur == d：高位 0..high-1 低位任意（high * p），高位取 high 时低位 0..low
        （low + 1），贡献 high * p + low + 1；
      - cur < d：只有高位 0..high-1，贡献 high * p。

    d == 0 时要额外小心「前导零」：最高位不能靠补 0 凑数，所以 high == 0 的位
    直接跳过；且因为把 0 当成「这一位就是 0」会重复计数，cur == 0 时高位要从
    1 开始数，即 high-1 种。

复杂度：时间 O(log high)，空间 O(1)。
"""


def _count_upto(n, d):
    if n <= 0:
        return 0
    count = 0
    p = 1
    while p <= n:
        high = n // (p * 10)
        cur = (n // p) % 10
        low = n % p
        if d != 0:
            if cur > d:
                count += (high + 1) * p
            elif cur == d:
                count += high * p + low + 1
            else:
                count += high * p
        else:
            if high > 0:
                if cur == 0:
                    count += (high - 1) * p + low + 1
                else:
                    count += high * p
        p *= 10
    return count


def digit_count_in_range(d, low, high):
    return _count_upto(high, d) - _count_upto(low - 1, d)


if __name__ == "__main__":
    assert digit_count_in_range(1, 1, 13) == 6
    assert digit_count_in_range(3, 100, 250) == 35
    assert digit_count_in_range(0, 1, 9) == 0
    assert digit_count_in_range(0, 1, 99) == 9
    assert digit_count_in_range(0, 1, 100) == 11
    assert digit_count_in_range(2, 1, 22) == 6
    assert digit_count_in_range(9, 1, 1000000) == 600000
    print("digit_count_in_range: all tests passed")
