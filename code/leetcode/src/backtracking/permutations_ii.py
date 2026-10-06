"""47. 全排列 II（Permutations II）

题目：给定一个可包含重复数字的序列 nums，按任意顺序返回所有不重复的全排列。

思路（排序 + used + 同层去重）：
    在 46 题的基础上多了「元素可能重复」，直接跑会产出重复排列。解决办法和
    90 题同源：先排序，让相同的数相邻，再用 used 记录已选，并加一条去重规则：

        若 i > 0 且 nums[i] == nums[i-1] 且 nums[i-1] 还没被使用，
        则跳过 nums[i]。

    这条规则的含义是：在同一个「坑位」上，相同数值只让第一个上。
    为什么要求「前一个相同数未被使用」？因为如果 nums[i-1] 已经被用了，
    说明它属于当前路径的上层（比如 [1,1] 里的两个 1 先后放），
    这时再放 nums[i] 是合法且必要的；只有当 nums[i-1] 闲置、
    却被我们越过时，才说明本轮在同层重复选值，必须剪掉。

复杂度：时间 O(n·n!)，空间 O(n)（递归深度 + used，不含答案本身）。
"""


def permute_unique(nums):
    nums = sorted(nums)
    res = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res


if __name__ == "__main__":
    out = permute_unique([1, 1, 2])
    assert len(out) == 3
    assert [1, 1, 2] in out
    assert [1, 2, 1] in out
    assert [2, 1, 1] in out

    assert permute_unique([1]) == [[1]]
    assert permute_unique([1, 1, 1]) == [[1, 1, 1]]
    print("permute_unique: all tests passed")
