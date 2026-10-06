"""461. 汉明距离（Hamming Distance）

题目：两个整数之间的汉明距离指它们二进制表示中不同位的个数。给定 x、y，求距离。

思路（异或 + 数 1）：
    异或的每一位规则是「相同为 0、不同为 1」，所以 x ^ y 的二进制里，1 出现的位置正好
    就是 x、y 不同的位。于是问题变成「求 x ^ y 中 1 的个数」，直接调用 191 的
    hamming_weight（n & (n - 1) 清最低位 1）即可。

复杂度：时间 O(k)（k = 不同位的个数，最坏 O(32)），空间 O(1)。
"""


def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


def hamming_distance(x, y):
    return hamming_weight(x ^ y)


if __name__ == "__main__":
    assert hamming_distance(1, 4) == 2
    assert hamming_distance(3, 1) == 1
    assert hamming_distance(0, 0) == 0
    assert hamming_distance(0, 0xFFFFFFFF) == 32
    assert hamming_distance(7, 7) == 0
    print("hamming_distance: all tests passed")
