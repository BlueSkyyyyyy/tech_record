"""421. 数组中两个数的最大异或值

题目：给定整数数组 nums，返回 nums[i] XOR nums[j] 的最大值（0 <= i, j < n）。

思路：把每个数的二进制表示（从最高位到最低位）插进一棵 0/1 字典树。求答案时，
      对每个数 num 从最高位往下走：在每一位上，若树里存在与 num 当前位**相反**的
      孩子，就走过去，并把答案的这一位记成 1；否则只能走相同的位，这一位为 0。

      为什么「能选相反位就一定要选」：异或结果在某一位是 1，当且仅当两个数该位不同。
      从最高位开始贪心，越高的位对数值贡献越大（2 的幂），只要当前位存在相反的分支，
      选它得到的数一定比放弃它更大，无论低位怎么走。这是按位贪心的标准论证。

      为什么用字典树而不是两两枚举：枚举是 O(n^2)。字典树把「和某个数异或最大的
      搭档」变成一次「每层尽量走反方向」的 O(位数) 查询，总复杂度降到 O(n·位数)。

复杂度：O(n·B)，B 为位数（本题取 31 位）。空间 O(n·B)。
"""

BITS = 31


def find_maximum_xor(nums):
    trie = {}
    for num in nums:
        node = trie
        for i in range(BITS, -1, -1):
            bit = (num >> i) & 1
            node = node.setdefault(bit, {})

    best = 0
    for num in nums:
        node = trie
        current = 0
        for i in range(BITS, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            if want in node:
                current |= 1 << i
                node = node[want]
            else:
                node = node[bit]
        best = max(best, current)
    return best


if __name__ == "__main__":
    assert find_maximum_xor([3, 10, 5, 25, 2, 8]) == 28
    assert find_maximum_xor([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]) == 127
    assert find_maximum_xor([0]) == 0
    assert find_maximum_xor([2, 4]) == 6
    assert find_maximum_xor([8, 10, 2]) == 10
    print("maximum_xor: all tests passed")
