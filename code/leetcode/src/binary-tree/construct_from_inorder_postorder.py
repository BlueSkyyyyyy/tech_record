"""106. 从中序与后序遍历序列构造二叉树

题目：给定一棵树的中序遍历 inorder 与后序遍历 postorder（无重复元素），
构造出这棵二叉树并返回根节点。

思路（分治：用后序定根、中序分左右）：
    这是 105 的「镜像」版本。后序的顺序是「左子树 → 右子树 → 根」，
    所以后序的**最后一个**元素一定是整棵树的根。
    知道根后，在中序里定位它：左边是左子树节点、右边是右子树节点。

    再回到后序：开头的「左子树大小」个元素是左子树的后序，
    紧接着的「右子树大小」个是右子树的后序，最后一位是根。
    对两段递归即可。同样用「值 → 中序下标」哈希表 + 下标区间，
    避免切数组，整体 O(n)。

    和 105 对照记忆：前序从**前**取根、后序从**后**取根，
    两者都是「用知根的那一趟定根、用中序分量」，本质同一套分治。

复杂度：时间 O(n)，空间 O(n)（哈希表与递归栈）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree_in_post(inorder, postorder):
    index = {value: i for i, value in enumerate(inorder)}

    def build(in_lo, in_hi, post_lo, post_hi):
        if in_lo > in_hi:
            return None
        root_val = postorder[post_hi]
        root = TreeNode(root_val)
        mid = index[root_val]
        left_size = mid - in_lo
        root.left = build(in_lo, mid - 1, post_lo, post_lo + left_size - 1)
        root.right = build(mid + 1, in_hi, post_lo + left_size, post_hi - 1)
        return root

    return build(0, len(inorder) - 1, 0, len(postorder) - 1)


def inorder_of(node):
    if node is None:
        return []
    return inorder_of(node.left) + [node.val] + inorder_of(node.right)


def postorder_of(node):
    if node is None:
        return []
    return postorder_of(node.left) + postorder_of(node.right) + [node.val]


if __name__ == "__main__":
    cases = [
        ([], []),
        ([1], [1]),
        ([9, 3, 15, 20, 7], [9, 15, 7, 20, 3]),
        ([-1], [-1]),
        ([4, 2, 5, 1, 3], [4, 5, 2, 3, 1]),
    ]
    for ino, post in cases:
        tree = build_tree_in_post(ino, post)
        assert inorder_of(tree) == ino
        assert postorder_of(tree) == post
    print("construct_from_inorder_postorder: all tests passed")
