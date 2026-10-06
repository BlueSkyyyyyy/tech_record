"""371. 两整数之和（Sum of Two Integers）

题目：不用运算符 + 和 -，计算两个整数 a、b 之和。

思路（异或当无进位加法，与运算进位移位当进位）：
    先只看「不加进位」的加法：a ^ b 在同一位置上，1+1 得 0、1+0 得 1，正好是不进位
    的结果；而需要进的位由 a & b 标出——两个都为 1 的位置才产生进位，进位要左移一位
    加到更高位上。于是反复做：
        carry = (a & b) << 1
        a = a ^ b          # 无进位和
        b = carry          # 把进位当作新的加数继续加
    直到进位 b 变成 0，此时的 a 就是最终和。

    Python 整数是任意精度、负数按无穷位补码，直接循环进位会无限增长，所以用一个 32 位
    掩码把中间结果截断，最后再把结果从「32 位无符号」还原成有符号整数。

复杂度：时间 O(32) == O(1)（最坏每次消掉一组进位），空间 O(1)。
"""


def get_sum(a, b):
    mask = 0xFFFFFFFF
    while b:
        carry = ((a & b) << 1) & mask
        a = (a ^ b) & mask
        b = carry
    return a if a <= 0x7FFFFFFF else ~(a ^ mask)


if __name__ == "__main__":
    assert get_sum(1, 2) == 3
    assert get_sum(2, 3) == 5
    assert get_sum(-1, 1) == 0
    assert get_sum(-2, 3) == 1
    assert get_sum(-5, -7) == -12
    assert get_sum(0, 0) == 0
    assert get_sum(123, 456) == 579
    print("get_sum: all tests passed")
