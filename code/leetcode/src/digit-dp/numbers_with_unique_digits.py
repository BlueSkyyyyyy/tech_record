"""357. 统计各位数字都不同的数字个数（Count Numbers with Unique Digits）

题目：给定 n，统计 [0, 10^n) 内各位数字都不同的整数个数。

思路（按「长度」用乘法原理计数，比数位 DP 更直接）：
    位数不同，选法天然分层：

    - 长度 0：只有 0 自己，1 个；
    - 长度 1：0..9，10 个；
    - 长度 L >= 2：首位不能是 0（9 种），第二位可选剩余的 9 个数字（含 0），
      第三位可选剩余 8 个，…… 第 L 位可选剩余 (10 - L + 1) 个。

    所以长度 L 的个数是 9 * 9 * 8 * ...，把长度 1..n 累加即可。
    当 L > 10 时不可能再不同（鸽巢原理），乘积自然变成 0，循环不会多算。

    这类「位数有限、每选一位就少一个可用数字」的计数，本质就是数位 DP 的
    闭式解——把「当前已用哪些数字」的掩码状态化简成了「还剩几个数字」。

复杂度：时间 O(min(n, 10))，空间 O(1)。
"""


def count_numbers_with_unique_digits(n):
    if n == 0:
        return 1
    total = 10
    cur = 9
    for length in range(2, n + 1):
        cur *= 10 - length + 1
        total += cur
    return total


if __name__ == "__main__":
    assert count_numbers_with_unique_digits(0) == 1
    assert count_numbers_with_unique_digits(1) == 10
    assert count_numbers_with_unique_digits(2) == 91
    assert count_numbers_with_unique_digits(3) == 739
    assert count_numbers_with_unique_digits(11) == 8877691
    print("count_numbers_with_unique_digits: all tests passed")
