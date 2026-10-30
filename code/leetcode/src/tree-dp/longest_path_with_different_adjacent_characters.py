"""2246. 相邻字符不同的最长路径（Longest Path With Different Adjacent Characters）

题目：给定一棵以 0 为根的树（parent[i] 是节点 i 的父节点，parent[0] = -1），
以及每个节点上的字符 s[i]。求一条路径，使路径上相邻节点的字符都不同，返回最长路径的节点数。

思路（树形 DP：孩子数组 + 取最大的两条臂）：
    这是「二叉树最大路径和 / 最长同值路径」在**一般树**上的版本。区别有两个：
      1. 孩子不再只有两个，要用孩子数组遍历（或按 parent 建邻接表）；
      2. 长度按**节点数**算，单条臂的起点节点也要计入。
    dfs 返回「以当前节点为起点、向下延伸的最长合法单链节点数」：
        arm = 1 + max(孩子 dfs 值，若 s[孩子] != s[当前])
    在返回之前，把当前节点能接的**最长两条臂**拼起来（两条臂都合法时路径可以拐过当前
    节点），得到 `top1 + top2 + 1` 个节点，用它更新全局最优。
    只能拼两条：路径在当前节点处最多分叉一次。

复杂度：时间 O(n)，空间 O(n)（孩子数组 + 递归栈 O(h)）。
"""

from collections import deque


def build_children(parent):
    children = [[] for _ in range(len(parent))]
    for node in range(1, len(parent)):
        children[parent[node]].append(node)
    return children


def longest_path(parent, s):
    children = build_children(parent)
    best = 0

    def dfs(node):
        nonlocal best
        top1 = top2 = 0
        for child in children[node]:
            arm = dfs(child)
            if s[child] == s[node]:
                continue
            if arm > top1:
                top1, top2 = arm, top1
            elif arm > top2:
                top2 = arm
        best = max(best, top1 + top2 + 1)
        return top1 + 1

    dfs(0)
    return best


if __name__ == "__main__":
    assert longest_path([-1, 0, 0, 1, 1, 2], "abacbe") == 3
    assert longest_path([-1, 0, 0, 0], "aabc") == 3
    assert longest_path([-1], "z") == 1
    assert longest_path([-1, 0, 1, 2], "abcd") == 4
    print("longest_path_with_different_adjacent_characters: all tests passed")
