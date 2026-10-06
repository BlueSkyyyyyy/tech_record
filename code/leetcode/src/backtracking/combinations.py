"""77. 组合（Combinations）

题目：给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
可以按任意顺序返回答案。

思路（回溯 + start 控制不回头）：
    组合和子集是同一套决策树，区别只在「叶子」的判定：子集在每个节点都收集，
    组合则只在凑够 k 个数（path 长度为 k）时才收集。

    同样用 start 保证「只能从当前数往后选」，这样 [1,4] 和 [4,1] 只会出现一次。
    为了效率，循环上界还能剪一刀：如果剩下可选的数全选上都不够 k 个，就不用
    再往下试了。当前 path 已有 len(path) 个，还差 k - len(path) 个；从 i 选到 n
    一共 n - i + 1 个数，要求它 ≥ 还差的数量，于是 i ≤ n - (k - len(path)) + 1。
    这一剪枝能砍掉整棵无用的子树。

复杂度：时间 O(C(n,k)·k)（共 C(n,k) 个组合，每个复制要 O(k)），
    空间 O(k)（递归深度，不含答案本身）。
"""


def combine(n, k):
    res = []
    path = []

    def backtrack(start):
        if len(path) == k:
            res.append(path[:])
            return
        need = k - len(path)
        for i in range(start, n - need + 2):
            path.append(i)
            backtrack(i + 1)
            path.pop()

    backtrack(1)
    return res


if __name__ == "__main__":
    assert combine(4, 2) == [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    assert combine(1, 1) == [[1]]
    assert combine(3, 3) == [[1, 2, 3]]
    assert combine(3, 4) == []
    print("combine: all tests passed")
