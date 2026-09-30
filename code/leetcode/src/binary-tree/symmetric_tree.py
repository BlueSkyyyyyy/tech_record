"""101. 对称二叉树（Symmetric Tree）

题目：给定一棵二叉树的根节点 root，判断它是否轴对称（左右互为镜像）。

思路（自顶向下比较两个节点，递归）：
    一棵树是否对称，取决于它的左右两棵子树是否互为镜像。
    于是问题变成比较两个节点 a、b 是否镜像：
      - 两个都为空 → 镜像是空的，成立；
      - 只有一个为空 → 不成立；
      - 值不同 → 不成立；
      - 值相同 → 继续比较「a 的左 vs b 的右」和「a 的右 vs b 的左」。
    注意第二条比较的是交叉的两对，而不是同侧的两对——这正是「镜像」的含义。

    为什么不能直接比较左右子树相等：
    「相等」要求 a 的左对 b 的左、a 的右对 b 的右；
    而「镜像」要求 a 的左对 b 的右、a 的右对 b 的左。两者在递归时走的方向不同。

复杂度：时间 O(n)（每个节点被比较一次），空间 O(h)（h 为树高，递归栈深度）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return (
            a.val == b.val
            and is_mirror(a.left, b.right)
            and is_mirror(a.right, b.left)
        )

    return root is None or is_mirror(root.left, root.right)


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
    assert is_symmetric(build_tree([])) is True
    assert is_symmetric(build_tree([1])) is True
    assert is_symmetric(build_tree([1, 2, 2])) is True
    assert is_symmetric(build_tree([1, 2, 2, 3, 4, 4, 3])) is True
    assert is_symmetric(build_tree([1, 2, 2, None, 3, None, 3])) is False
    assert is_symmetric(build_tree([1, 2, 2, 2, None, 2])) is False
    print("symmetric_tree: all tests passed")
