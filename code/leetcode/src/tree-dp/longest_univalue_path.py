"""687. 最长同值路径（Longest Univalue Path）

题目：给定二叉树，找最长的路径，路径上每个节点的值都相同。路径长度用「边的条数」衡量。

思路（树形 DP：后序 + 返回「以我为端点的单臂长度」）：
    一条同值路径在树上有一个「最高点」节点，它从该节点向左右各延伸一条同值单链。
    递归函数返回：从当前节点向下、能延伸的**单臂**同值边数（只能走一个方向）。
      - 若左孩子值与当前相同，则左臂长 = 左孩子返回的单臂长 + 1，否则为 0；
      - 右臂同理。
    以当前节点为「最高点」的路径 = 左臂 + 右臂，用它更新全局最优；
    但返回给父节点的只能是 max(左臂, 右臂)，因为父节点接过来时只能经过一条边。

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


def longest_univalue_path(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        left_arm = left + 1 if node.left and node.left.val == node.val else 0
        right_arm = right + 1 if node.right and node.right.val == node.val else 0
        best = max(best, left_arm + right_arm)
        return max(left_arm, right_arm)

    dfs(root)
    return best


if __name__ == "__main__":
    assert longest_univalue_path(build([5, 4, 5, 1, 1, None, 5])) == 2
    assert longest_univalue_path(build([1, 4, 5, 4, 4, None, 5])) == 2
    assert longest_univalue_path(build([1])) == 0
    assert longest_univalue_path(build([1, 1, 1, 1, 1, None, 1])) == 4
    print("longest_univalue_path: all tests passed")
