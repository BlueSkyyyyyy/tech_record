"""1372. 二叉树中的最长交错路径（Longest ZigZag Path in a Binary Tree）

题目：从任一节点出发向下走，每一步可以走左孩子或右孩子；要求相邻两步方向相反
（左-右-左… 或 右-左-右…）。求最长的交错路径长度（按边的条数算）。

思路（树形 DP：后序 + 返回「两个方向出发的最长交错长度」）：
    递归返回 `(go_left, go_right)`，含义是**从当前节点出发**、第一步分别走左/右时，
    能走出的最长交错边数：
      - go_left = 若左孩子存在，则 1 + 左孩子的 go_right（走左后下一步必须走右）；
      - go_right = 若右孩子存在，则 1 + 右孩子的 go_left。
    每个节点处用 max(go_left, go_right) 更新全局最优，因为最长路径的最高点可能不是根。

复杂度：时间 O(n)，空间 O(h)。
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(vals):
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root


def longest_zigzag(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (0, 0)
        left_go_left, left_go_right = dfs(node.left)
        right_go_left, right_go_right = dfs(node.right)
        go_left = 1 + left_go_right if node.left else 0
        go_right = 1 + right_go_left if node.right else 0
        best = max(best, go_left, go_right)
        return (go_left, go_right)

    dfs(root)
    return best


if __name__ == "__main__":
    big = [1, None, 1, 1, 1, None, None, 1, 1, None, 1, None, None,
           None, 1, None, 1]
    assert longest_zigzag(build(big)) == 3
    assert longest_zigzag(build([1, 1, 1, None, 1, None, None, 1, 1, None, 1])) == 4
    assert longest_zigzag(build([1])) == 0
    assert longest_zigzag(build([1, 2])) == 1
    print("longest_zigzag_path: all tests passed")
