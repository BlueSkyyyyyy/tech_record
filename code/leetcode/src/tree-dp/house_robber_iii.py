"""337. 打家劫舍 III（House Robber III）

题目：房屋排列成二叉树。不能同时偷「直接相连」的两个节点（父子）。
求能偷到的最大金额。

思路（树形 DP：后序 + 返回「选/不选」两个状态）：
    对每个节点，父节点的决策只依赖子节点「偷/不偷」这两种状态，因此递归函数
    返回一个二元组 `(rob, not_rob)`：
      - rob：以该节点为根的子树，且**偷该节点**时的最大金额；
      - not_rob：以该节点为根的子树，且**不偷该节点**时的最大金额。
    转移（后序，先算左右孩子）：
      - 偷自己 ⇒ 孩子都不能偷：node.val + left.not_rob + right.not_rob
      - 不偷自己 ⇒ 孩子可偷可不偷，各自取最大：
        max(left.rob, left.not_rob) + max(right.rob, right.not_rob)

复杂度：时间 O(n)（每个节点只访问一次），空间 O(h)（递归栈，h 为树高）。
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(vals):
    """按层序数组建树，None 表示空节点。"""
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


def rob(root):
    def dfs(node):
        if node is None:
            return (0, 0)
        l_rob, l_not = dfs(node.left)
        r_rob, r_not = dfs(node.right)
        rob_here = node.val + l_not + r_not
        not_here = max(l_rob, l_not) + max(r_rob, r_not)
        return (rob_here, not_here)

    return max(dfs(root))


if __name__ == "__main__":
    assert rob(build([3, 2, 3, None, 3, None, 1])) == 7
    assert rob(build([3, 4, 5, 1, 3, None, 1])) == 9
    assert rob(build([])) == 0
    assert rob(build([5])) == 5
    print("house_robber_iii: all tests passed")
