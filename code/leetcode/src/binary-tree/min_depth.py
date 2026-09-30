"""111. 二叉树的最小深度（Minimum Depth of Binary Tree）

题目：给定一棵二叉树的根节点 root，返回它的最小深度。
最小深度是从根节点到最近叶子节点的最短路径上的节点数。

思路（后序递归，但要先排除「单侧为空」）：
    直观想法是 `1 + min(左, 右)`，但这是错的。看这棵树：
        1
         \
          2
    根只有右孩子。左子树的深度按定义是 0，若直接取 min 会得到 1 + 0 = 1，
    可实际根不是叶子，它到最近叶子 2 的距离是 2。错误根源在于：
    「空子树」不是叶子，不能参与取 min。

    正确做法是先分流：
      - 只有左孩子为空：答案只能来自右子树，返回 1 + 右；
      - 只有右孩子为空：返回 1 + 左；
      - 两个孩子都在：才返回 1 + min(左, 右)；
      - 空节点返回 0。

    为什么最大深度不用这么麻烦：因为在 max 里 0 永远不可能是最大值（除非两边都空），
    空子树不会「污染」结果；而 min 会让空子树抢答，所以必须显式排除。

复杂度：时间 O(n)（每个节点一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def min_depth(root):
    if root is None:
        return 0
    if root.left is None:
        return 1 + min_depth(root.right)
    if root.right is None:
        return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))


def build_tree(values):
    if not values:
        return None
    root = TreeNode(values[0])
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


if __name__ == "__main__":
    assert min_depth(build_tree([])) == 0
    assert min_depth(build_tree([1])) == 1
    assert min_depth(build_tree([1, None, 2])) == 2
    assert min_depth(build_tree([3, 9, 20, None, None, 15, 7])) == 2
    assert min_depth(build_tree([1, 2, None, 3, None, 4])) == 4
    assert min_depth(build_tree([1, 2, 3, 4, None, 5, 6])) == 3
    print("min_depth: all tests passed")
