"""78. 子集（Subsets）· 位枚举法

题目：给定不含重复元素的整数数组 nums，返回它的所有子集（幂集）。

思路（一个整数当一张"选/不选"的开关表）：
    长度为 n 的数组，每个元素只有"选"和"不选"两种状态，一共 2^n 种组合。用一个
    从 0 到 2^n - 1 的整数 mask 表示一种组合：mask 的第 i 位为 1 表示选 nums[i]。
    枚举所有 mask，逐个翻译成子集即可。

    这是"状态压缩"的最小示例：当 n 不大时，用一个整数的二进制位就能表示一个集合，
    枚举整数就等价于枚举集合。它与回溯篇的 78 题是同一答案的两种视角：回溯是"逐个
    元素做决定"，位枚举是"直接列出所有决定的结果"。

复杂度：时间 O(n * 2^n)（2^n 种组合，每种花 O(n) 组装），空间 O(n * 2^n)（结果本身）。
"""


def subsets(nums):
    n = len(nums)
    res = []
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        res.append(subset)
    return res


if __name__ == "__main__":
    got = subsets([1, 2, 3])
    want = [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    assert len(got) == 8
    assert [sorted(s) for s in got] == [sorted(s) for s in want]
    assert subsets([]) == [[]]
    assert subsets([0]) == [[], [0]]
    print("subsets: all tests passed")
