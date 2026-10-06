"""338. 比特位计数（Counting Bits）

题目：给定整数 n，返回长度为 n + 1 的数组 ans，其中 ans[i] 是 i 的二进制表示中
「1」的个数（i 从 0 到 n）。

思路（位运算递推，不重复计算）：
    如果对每个 i 都单独数一遍 1，复杂度是 O(n log n)。但相邻整数的二进制有很强的
    联系：把 i 右移一位（i >> 1）相当于扔掉最低位。于是

        ans[i] = ans[i >> 1] + (i & 1)

    其中 i & 1 取的是被扔掉的最低位。也就是「i 的 1 的个数 = 去掉最低位后的数的
    1 的个数，再加上最低位本身是不是 1」。

    为什么从 0 到 n 正序递推是对的：i >> 1 一定小于 i（除 i = 0 外），也就是递推
    依赖的状态在数组里已经算好了，一遍正序循环即可，无需递归。

    另一种等价写法：ans[i] = ans[i & (i - 1)] + 1。因为 i & (i - 1) 会清掉 i 最低
    位的那个 1（见 191 题），所以它比 i 少一个 1，直接加 1 就得到 ans[i]。两种写法
    都把 O(log i) 的逐位统计降成了 O(1) 的查表。

复杂度：时间 O(n)（每个 i 只做常数次位运算），空间 O(n)（返回数组本身）。
"""


def count_bits(n):
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (i & 1)
    return ans


if __name__ == "__main__":
    assert count_bits(0) == [0]
    assert count_bits(1) == [0, 1]
    assert count_bits(2) == [0, 1, 1]
    assert count_bits(5) == [0, 1, 1, 2, 1, 2]
    assert count_bits(8) == [0, 1, 1, 2, 1, 2, 2, 3, 1]
    assert sum(count_bits(16)) == 33
    print("count_bits: all tests passed")
