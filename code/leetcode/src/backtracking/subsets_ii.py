"""90. 子集 II（Subsets II）

题目：给你一个整数数组 nums，其中可能包含重复元素，返回该数组所有可能的
子集（幂集）。解集不能包含重复的子集，可以按任意顺序返回。

思路（排序 + 同层去重）：
    如果直接套 78 题「每个节点都收集」的模板，碰到重复元素就会产出重复子集：
    比如 [2, 2] 里先选第一个 2 还是第二个 2，会得到两个一模一样的 [2]。

    关键观察：重复只可能出现在「同一层里选了数值相同的不同下标」。
    于是先给数组排序，让相同的数挨在一起，然后在每层循环里加一句：
    若当前下标 i 不是本层的起点，且 nums[i] == nums[i-1]，就跳过。
    这样每一层对同一个数值只用一次，重复分支被整棵剪掉。

    为什么是排序 + 「同层」判定而不是全局哈希：同一层代表「当前这一步选谁」，
    同层选到相同数值才会撞出重复；不同层选相同数值（如 [2, 2]）是允许的，
    所以判据必须带上 i > start 这个「不是本层第一个」的条件。

复杂度：时间 O(n·2^n)，空间 O(n)（递归深度，不含答案本身）。
"""


def subsets_with_dup(nums):
    nums = sorted(nums)
    res = []
    path = []

    def backtrack(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res


if __name__ == "__main__":
    out = subsets_with_dup([1, 2, 2])
    want = [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
    assert out == want

    assert subsets_with_dup([0]) == [[], [0]]
    assert subsets_with_dup([1, 1, 1]) == [[], [1], [1, 1], [1, 1, 1]]
    print("subsets_with_dup: all tests passed")
