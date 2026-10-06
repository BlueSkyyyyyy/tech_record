"""78. 子集（Subsets）

题目：给你一个整数数组 nums，数组中的元素互不相同，返回该数组所有可能的子集
（幂集）。解集不能包含重复的子集，可以按任意顺序返回。

思路（回溯 / 决策树，每个节点都是一种子集）：
    把「构造一个子集」想成一棵决策树：从下标 0 开始，每一步决定「要不要把
    当前位置的数放进当前子集」。树上的每个节点（包括根）都对应一个合法子集，
    所以每进入一次递归就把 path 记进答案。

    为了让同一组合只出现一次（比如 [1,2] 与 [2,1] 视为同一个子集），
    我们规定「只能从当前下标往后挑」，用参数 start 控制可选范围：
    选了 nums[i] 后，下一层只能从 i+1 开始，天然避免了重复和乱序。

    为什么不需要 used 数组：因为不允许回头选，每个元素在一层里只用一次，
    下标递增本身就把「已经用过的」挡在了范围之外。

复杂度：时间 O(n·2^n)（共 2^n 个子集，每个子集复制进答案要 O(n)），
    空间 O(n)（递归深度，不含答案本身）。
"""


def subsets(nums):
    res = []
    path = []

    def backtrack(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res


if __name__ == "__main__":
    out = subsets([1, 2, 3])
    assert len(out) == 8
    assert [] in out
    assert [1, 2, 3] in out
    assert [1, 3] in out
    assert [3, 1] not in out

    assert subsets([0]) == [[], [0]]
    assert subsets([]) == [[]]
    print("subsets: all tests passed")
