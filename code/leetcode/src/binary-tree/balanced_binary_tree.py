"""110. 平衡二叉树（Balanced Binary Tree）

题目：给定一棵二叉树的根节点 root，判断它是否是高度平衡的。
高度平衡指：每个节点的左右两棵子树的高度差都不超过 1。

思路（后序递归：一边求高度，一边判平衡）：
    直观做法是对每个节点调用一次「求高度」，再判断两子树高度差。
    但那样每个节点会被重复求高度，最坏 O(n^2)。
    更好的做法是让递归函数**同时**完成两件事：返回子树高度，顺便检查该子树是否平衡。

    约定递归函数返回「高度」，若发现不平衡就返回 -1 作为标记：
      - 空节点高度 0；
      - 先拿左子树高度，若为 -1 直接返回 -1（短路）；
      - 再拿右子树高度，若为 -1 直接返回 -1；
      - 若 |左高 - 右高| > 1，返回 -1；
      - 否则返回 1 + max(左高, 右高)。
    最终答案就是「返回值不等于 -1」。

    为什么能一边求高一边判断：判断一个节点是否平衡，正好需要左右子树的
    高度；而计算这些高度本来就要递归子节点。把两件事合并到同一次递归里，
    每个节点只算一次，复杂度从 O(n^2) 降到 O(n)。这是「顺路更新信息」的典型。

复杂度：时间 O(n)（每个节点一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_balanced(root):
    def height(node):
        if node is None:
            return 0
        left = height(node.left)
        if left == -1:
            return -1
        right = height(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return height(root) != -1


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
    assert is_balanced(build_tree([])) is True
    assert is_balanced(build_tree([1])) is True
    assert is_balanced(build_tree([3, 9, 20, None, None, 15, 7])) is True
    assert is_balanced(build_tree([1, 2, 2, 3, 3, None, None, 4, 4])) is False
    assert is_balanced(build_tree([1, 2, None, 3, None, 4])) is False
    assert is_balanced(build_tree([1, 2, 3])) is True
    print("balanced_binary_tree: all tests passed")
