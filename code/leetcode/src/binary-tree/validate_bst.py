"""98. 验证二叉搜索树（Validate Binary Search Tree）

题目：给定一棵二叉树的根节点 root，判断它是否是一棵有效的二叉搜索树（BST）。
BST 定义：任意节点的左子树所有值都**小于**它，右子树所有值都**大于**它。

思路（递归时携带上下界）：
    只比较「父节点和直接孩子」是不够的。反例：
        5
       / \
      1   6
         / \
        3   7
    每个父子关系都满足「左小右大」，但 3 在 5 的右子树里却比 5 小，不是 BST。
    问题出在：约束不是来自直接父亲，而是来自**整条祖先链**。

    正确做法是递归时带上一个开区间 (low, high)：
      - 根节点的允许范围是 (-∞, +∞)；
      - 进入左子树时把上界收紧为当前节点的值，即 (low, node.val)；
      - 进入右子树时把下界收紧为当前节点的值，即 (node.val, high)；
      - 当前节点的值必须严格落在 (low, high) 内，否则不是 BST。
    每个节点只被检查一次，天然满足「祖先链上的所有约束」。

    为什么等价于中序有序：中序遍历 BST 会得到严格递增序列，
    「值始终落在由祖先收紧的区间内」正是这个递增性的递归表述。

复杂度：时间 O(n)（每个节点一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def is_valid_bst(root):
    def check(node, low, high):
        if node is None:
            return True
        if not (low < node.val < high):
            return False
        return check(node.left, low, node.val) and check(node.right, node.val, high)

    return check(root, float("-inf"), float("inf"))


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
    assert is_valid_bst(build_tree([])) is True
    assert is_valid_bst(build_tree([2, 1, 3])) is True
    assert is_valid_bst(build_tree([5, 1, 4, None, None, 3, 6])) is False
    assert is_valid_bst(build_tree([5, 4, 6, None, None, 3, 7])) is False
    assert is_valid_bst(build_tree([2, 2, 2])) is False
    assert is_valid_bst(build_tree([-2147483648])) is True
    print("validate_bst: all tests passed")
