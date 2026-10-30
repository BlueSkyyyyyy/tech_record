"""1026. 节点与其祖先之间的最大差值（Maximum Difference Between Node and Ancestor）

题目：给定二叉树，对任意一对「祖先 a、后代 b」，求 |a.val - b.val| 的最大值。

思路（后序返回子树极值）：
    每对祖先—后代都唯一对应「某个节点是另一个节点的祖先」。与其自上而下传祖先，
    不如自下而上：让 dfs 返回**当前子树的最小值和最大值**，回到当前节点时，它一定
    是子树里所有节点的祖先，于是
        max(|node.val - 子树最小值|, |node.val - 子树最大值|)
    就是以当前节点为祖先时的最大差值。每个节点处结算一次，取全局最大即可。

    为什么这样不漏：任意一对 (祖先 u, 后代 v)，在递归回到 u 时 v 还在 u 的子树里，
    u 的子树极值一定覆盖了 v.val，所以这组差一定被算到。

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


def max_ancestor_diff(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (float("inf"), float("-inf"))
        l_min, l_max = dfs(node.left)
        r_min, r_max = dfs(node.right)
        low = min(l_min, r_min, node.val)
        high = max(l_max, r_max, node.val)
        best = max(best, abs(node.val - low), abs(node.val - high))
        return (low, high)

    dfs(root)
    return best


if __name__ == "__main__":
    assert max_ancestor_diff(build([8, 3, 10, 1, 6, None, 14, None, None, 4, 7, 13])) == 7
    assert max_ancestor_diff(build([1, None, 2, None, 0, 3])) == 3
    assert max_ancestor_diff(build([1])) == 0
    assert max_ancestor_diff(build([4, 2, 6, 1, 3, 5, 7])) == 3
    print("maximum_difference_between_node_and_ancestor: all tests passed")
