"""201. 数字范围按位与（Bitwise AND of Numbers Range）

题目：给定两个整数 left、right（left <= right），返回区间 [left, right] 内所有整数
按位与的结果。

思路（找公共前缀）：
    区间内相邻整数不断加 1，低位会不停翻转：只要一个位在区间里既出现过 0 又出现过 1，
    那么按位与之后它必然变成 0。最终能保留下来的，只有所有数都相同的那一段——也就是
    left 和 right 的**公共二进制前缀**。

    于是不断把 left、right 一起右移，直到两者相等（此时已经把不同的低位全部移掉），
    记下右移的次数 shift，再把 left 左移 shift 位补回零，就是答案。

    例：left=5(101), right=7(111)。右移一次变 2(10) 和 3(11)，不等；再右移变 1(1) 和
    1(1)，相等。shift=2，答案 1 << 2 = 4(100)。验证 5&6&7 = 4，正确。

复杂度：时间 O(32) == O(1)，空间 O(1)。
"""


def range_bitwise_and(left, right):
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift


if __name__ == "__main__":
    assert range_bitwise_and(5, 7) == 4
    assert range_bitwise_and(0, 0) == 0
    assert range_bitwise_and(1, 1) == 1
    assert range_bitwise_and(0, 1) == 0
    assert range_bitwise_and(1, 2) == 0
    assert range_bitwise_and(10, 10) == 10
    assert range_bitwise_and(1, 3) == 0
    assert range_bitwise_and(4, 7) == 4
    print("range_bitwise_and: all tests passed")
