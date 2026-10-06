"""46. 全排列（Permutations）

题目：给定一个不含重复数字的数组 nums，返回其所有可能的全排列。
可以按任意顺序返回答案。

思路（回溯 + used 标记，顺序有关）：
    排列和组合的关键差别是「顺序有意义」：[1,2] 和 [2,1] 是两个不同的排列。
    所以不能再靠 start「只往后选」来去重，而是每一层都从下标 0 开始重新挑，
    用一个布尔数组 used 记录哪些位置已经被用过，跳过用过的即可。

    决策树：每一层决定「下一个位置放哪个还没用过的数」。当 path 的长度等于
    n 时，说明每个数都用上了，得到一个完整排列，收进答案后回溯。

    为什么 used 必须对称地「用后恢复」：path 和 used 是同一份状态的两个侧面，
    选一个数就同时改两者，回溯时也要同时还原，否则后面的分支会带着脏状态。

复杂度：时间 O(n·n!)（共 n! 个排列，每个复制长度 n），
    空间 O(n)（递归深度 + used，不含答案本身）。
"""


def permute(nums):
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
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res


if __name__ == "__main__":
    out = permute([1, 2, 3])
    assert len(out) == 6
    assert [1, 2, 3] in out
    assert [3, 2, 1] in out
    assert [1, 1, 2] not in out

    assert permute([1]) == [[1]]
    assert permute([]) == [[]]
    print("permute: all tests passed")
