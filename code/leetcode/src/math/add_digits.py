"""258. 各位相加（Add Digits）

题目：给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数，返回该结果。

思路（数字根公式 O(1) 求解）：
    反复各位相加的最终结果叫做「数字根」，它等于 num 对 9 取模（余数为 0 时取 9）：
        return 0 if num == 0 else 1 + (num - 1) % 9

    为什么数字根与模 9 有关：一个数和它的各位数字之和对 9 同余。因为
    10 ≡ 1 (mod 9)，所以 10^k ≡ 1 (mod 9)，一个数 d_k...d_0 与各位之和
    d_k + ... + d_0 模 9 相等。反复各位相加不改变其余数，最终停在一位数上，那一
    位数就是 num mod 9（在 1~9 之间）。唯一的例外是 num 本身是 9 的倍数时余数为 0，
    此时数字根取 9。

    为什么写成 `1 + (num - 1) % 9` 而不是 `num % 9 or 9`：前者把 9 的倍数、以及
    非 9 倍数两种情形统一到一条公式里，避免了 `or` 在 num = 0 时的歧义（0 的数字根
    应为 0）。

    为什么不用模拟：模拟每次 O(位数)，虽然也很快，但公式把它降到 O(1)，是这道题真正
    想考的「数学结论」。

复杂度：时间 O(1)，空间 O(1)。
"""


def add_digits(num):
    if num == 0:
        return 0
    return 1 + (num - 1) % 9


if __name__ == "__main__":
    assert add_digits(38) == 2
    assert add_digits(0) == 0
    assert add_digits(1) == 1
    assert add_digits(9) == 9
    assert add_digits(18) == 9
    assert add_digits(10) == 1
    assert add_digits(99) == 9
    assert add_digits(999) == 9
    assert add_digits(12345) == 6
    print("add_digits: all tests passed")
