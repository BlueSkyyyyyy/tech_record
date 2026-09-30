"""236. 二叉树的最近公共祖先（Lowest Common Ancestor of a Binary Tree）

题目：给定一棵二叉树和两个节点 p、q，返回它们的最近公共祖先（LCA）。
最近公共祖先指：所有同时是 p 和 q 祖先的节点中，深度最大的那个。
一个节点也可以是它自己的祖先。

思路（后序递归，按「子树里找到了谁」返回）：
    定义递归函数 `dfs(node)`：在以 node 为根的子树里查找 p 和 q，返回值含义：
      - 若 node 为空 → 返回空；
      - 若 node 就是 p 或 q → 返回 node（找到了其中一个）；
      - 否则递归左右子树，拿到 left、right：
          · left 和 right 都非空 → 说明 p、q 分居两侧，node 就是它们的 LCA；
          · 只有一侧非空 → 这一侧的结果继续往上返回；
          · 都为空 → 返回空。

    为什么这样能得到最近公共祖先：p、q 要么分居某个节点的左右两侧
    （该节点就是 LCA），要么其中一个就是另一个的祖先（此时先遇到的那个直接返回）。
    递归自下而上把「找到了谁」层层上报，第一个同时收到两侧结果的节点就是答案。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left is not None and right is not None:
        return root
    return left if left is not None else right


if __name__ == "__main__":
    n3 = TreeNode(3)
    n5 = TreeNode(5)
    n1 = TreeNode(1)
    n6 = TreeNode(6)
    n2 = TreeNode(2)
    n0 = TreeNode(0)
    n8 = TreeNode(8)
    n7 = TreeNode(7)
    n4 = TreeNode(4)
    n3.left, n3.right = n5, n1
    n5.left, n5.right = n6, n2
    n1.left, n1.right = n0, n8
    n2.left, n2.right = n7, n4

    assert lowest_common_ancestor(n3, n5, n1) is n3
    assert lowest_common_ancestor(n3, n5, n4) is n5
    assert lowest_common_ancestor(n3, n6, n4) is n5
    assert lowest_common_ancestor(n3, n7, n4) is n2
    assert lowest_common_ancestor(n3, n7, n7) is n7
    assert lowest_common_ancestor(n3, n0, n8) is n1
    print("lowest_common_ancestor: all tests passed")
