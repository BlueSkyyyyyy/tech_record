"""190. 颠倒二进制位（Reverse Bits）

题目：给定一个 32 位无符号整数 n，返回把它二进制位上下颠倒后的结果。例如输入
0000...000101（5），输出 10100000...0000（一个很大的数）。

思路（逐位搬移）：
    从 n 的最低位开始，一位一位地取出来，放到结果的最高位上。做 32 次：
      - ans 左移一位，给「下一位」腾出最低位；
      - 把 n 当前最低位（n & 1）放到 ans 的最低位；
      - n 右移一位，处理下一位。
    循环结束后 ans 就是颠倒后的结果。本质是把「取 n 的第 i 位」搬到「ans 的第 31-i 位」。

    关键细节：Python 的整数是任意精度的，n 可能是「负的补码语义」或位数不受限，所以
    先 n &= 0xFFFFFFFF 截成 32 位无符号，最后再对结果取同样掩码，保证只保留 32 位。

复杂度：时间 O(32) == O(1)（固定循环 32 次），空间 O(1)。
"""


def reverse_bits(n):
    n &= 0xFFFFFFFF
    ans = 0
    for _ in range(32):
        ans = ((ans << 1) | (n & 1)) & 0xFFFFFFFF
        n >>= 1
    return ans


if __name__ == "__main__":
    assert reverse_bits(0b00000010100101000001111010011100) == 0b00111001011110000010100101000000
    assert reverse_bits(0) == 0
    assert reverse_bits(1) == 0x80000000
    assert reverse_bits(0xFFFFFFFF) == 0xFFFFFFFF
    assert reverse_bits(2) == 0x40000000
    print("reverse_bits: all tests passed")
