"""1373. 二叉搜索子树的最大键值和（Maximum Sum BST in Binary Tree）

题目：给定二叉树，找出任意一棵「二叉搜索子树」（BST），使它的节点值之和最大，返回该最大和。
若不存在非空 BST 子树，返回 0（单节点天然是 BST，所以只要树非空答案就 ≥ 该节点值，除非全为负）。

思路（树形 DP：后序 + 返回四元组）：
    判断「以当前节点为根的子树是不是 BST」需要子树的四个信息：
      `(is_bst, min_val, max_val, sum_val)`。
    后序拿到左右子树信息后，当前子树是 BST 当且仅当：
      左是 BST、右是 BST，且 `左.max < node.val < 右.min`。
    - 成立：和 = 左和 + 右和 + node.val，更新全局最大值；
      返回 `(True, min(左.min, node.val), max(右.max, node.val), 和)`。
    - 不成立：整棵子树不是 BST，返回 `(False, ...)`，其聚合值不会被父节点采用。
    空节点返回 `(True, +∞, -∞, 0)`，这样任意单节点都能满足上面的区间条件。

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


def max_sum_bst(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (True, float("inf"), float("-inf"), 0)
        l_ok, l_min, l_max, l_sum = dfs(node.left)
        r_ok, r_min, r_max, r_sum = dfs(node.right)
        if l_ok and r_ok and l_max < node.val < r_min:
            total = l_sum + r_sum + node.val
            best = max(best, total)
            return (True, min(l_min, node.val), max(r_max, node.val), total)
        return (False, 0, 0, 0)

    dfs(root)
    return best


if __name__ == "__main__":
    assert max_sum_bst(build([1, 4, 3, 2, 4, 2, 5, None, None, None, None,
                              None, None, 4, 6])) == 20
    assert max_sum_bst(build([4, 3, None, 1, 2])) == 2
    assert max_sum_bst(build([-4, -2, -5])) == 0
    assert max_sum_bst(build([2, 1, 3])) == 6
    print("max_sum_bst: all tests passed")
