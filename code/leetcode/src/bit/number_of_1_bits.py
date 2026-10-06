"""191. 位 1 的个数（Number of 1 Bits）

题目：给定一个整数 n（视作无符号 32 位二进制），返回它二进制表示中「1」的个数，
也叫汉明重量（Hamming Weight）。

思路（n & (n - 1) 清掉最低位的 1）：
    每次执行 n = n & (n - 1)，都会把 n 二进制里最右边的一个 1 变成 0，其余的位不变。
    不断重复直到 n 变成 0，执行的次数就是 1 的个数。

    为什么 n & (n - 1) 能清掉最低位的 1：设 n 最低位的 1 在位置 k，则 n 的二进制形如
    `...1 0 0 ... 0`（第 k 位是 1，低位全是 0）。减 1 之后，第 k 位借位变成 0，所有
    更低的位翻转成 1，即变成 `...0 1 1 ... 1`。两者按位与：高位相同、第 k 位 1&0=0、
    低位的 0&1=0，于是第 k 位连同它下面的位全部归零，比它高的位原样保留。所以恰好清掉
    一个 1。

    另一种思路是逐位检查（每次右移一位、取最低位累加），需要固定 32 次；而本方法只在
    「1 的个数」次循环后退出，更省。对稀疏的二进制数（1 很少）尤其快。

    注意 Python 的整数是任意精度的，负数会用无穷位的补码语义，所以这里只按非负整数
    处理；若输入可能为负，可先把 n 与 0xFFFFFFFF 按位与截成 32 位无符号再看。

复杂度：时间 O(k)（k 是 1 的个数，最坏 O(32)），空间 O(1)。
"""


def hamming_weight(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


if __name__ == "__main__":
    assert hamming_weight(0) == 0
    assert hamming_weight(1) == 1
    assert hamming_weight(2) == 1
    assert hamming_weight(3) == 2
    assert hamming_weight(11) == 3
    assert hamming_weight(128) == 1
    assert hamming_weight(255) == 8
    assert hamming_weight(0xFFFFFFFF) == 32
    print("hamming_weight: all tests passed")
