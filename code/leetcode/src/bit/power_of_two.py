"""231. 2 的幂（Power of Two）

题目：给定一个整数 n，判断它是否是 2 的幂。

思路（清最低位的 1）：
    2 的幂的二进制表示里恰好只有一个 1（如 1、10、100、1000...）。而 n & (n - 1) 会
    清掉最低位的那个 1（见 number_of_1_bits.py）。所以：
      - n > 0 且 n & (n - 1) == 0  →  是 2 的幂。
    n = 0 时 n & (n - 1) 也是 0，所以必须加上 n > 0 这个前提。

    为什么负数要排除：负数最高位是 1，且按补码解释含有多个 1，不可能只有一个 1。

复杂度：时间 O(1)（一次位与），空间 O(1)。
"""


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


if __name__ == "__main__":
    assert is_power_of_two(1) is True
    assert is_power_of_two(2) is True
    assert is_power_of_two(4) is True
    assert is_power_of_two(16) is True
    assert is_power_of_two(3) is False
    assert is_power_of_two(0) is False
    assert is_power_of_two(-16) is False
    assert is_power_of_two(1 << 30) is True
    print("power_of_two: all tests passed")
