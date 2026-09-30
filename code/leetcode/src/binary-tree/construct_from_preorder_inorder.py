"""105. 从前序与中序遍历序列构造二叉树

题目：给定一棵树的前序遍历 preorder 与中序遍历 inorder（无重复元素），
构造出这棵二叉树并返回根节点。

思路（分治：用前序定根、中序分左右）：
    前序的顺序是「根 → 左子树 → 右子树」，所以前序的第一个元素一定是整棵树的根。
    中序的顺序是「左子树 → 根 → 右子树」，所以一旦知道根是谁，
    就能在中序里定位它：它左边的全部是左子树节点，右边的全部是右子树节点。

    拿到左右子树的节点集合后，回到前序里：
    紧跟根之后、长度为「左子树大小」的一段就是左子树的前序，
    再往后一段是右子树的前序。于是对两段各自递归，就能还原整棵树。

    实现上不真的切数组（那样每次 O(n) 会退化到 O(n^2)），
    而是用「下标区间」表示每段的边界，再用一个哈希表预先存好
    「值 → 中序下标」，定位根是 O(1)。这样每个节点只处理一次。

    为什么必须知道两种遍历：只有前序无法区分左右子树的分界，
    只有中序也无法确定谁在上层；前序给根、中序给分界，两者配合才唯一。
    同理 106 用「后序定根 + 中序分界」。

复杂度：时间 O(n)（建哈希表 + 每个节点一次），空间 O(n)（哈希表与递归栈）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree_pre_in(preorder, inorder):
    index = {value: i for i, value in enumerate(inorder)}

    def build(pre_lo, pre_hi, in_lo, in_hi):
        if pre_lo > pre_hi:
            return None
        root_val = preorder[pre_lo]
        root = TreeNode(root_val)
        mid = index[root_val]
        left_size = mid - in_lo
        root.left = build(pre_lo + 1, pre_lo + left_size, in_lo, mid - 1)
        root.right = build(pre_lo + left_size + 1, pre_hi, mid + 1, in_hi)
        return root

    return build(0, len(preorder) - 1, 0, len(inorder) - 1)


def preorder_of(node):
    if node is None:
        return []
    return [node.val] + preorder_of(node.left) + preorder_of(node.right)


def inorder_of(node):
    if node is None:
        return []
    return inorder_of(node.left) + [node.val] + inorder_of(node.right)


if __name__ == "__main__":
    cases = [
        ([], []),
        ([1], [1]),
        ([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]),
        ([-1], [-1]),
        ([1, 2, 4, 5, 3], [4, 2, 5, 1, 3]),
    ]
    for pre, ino in cases:
        tree = build_tree_pre_in(pre, ino)
        assert preorder_of(tree) == pre
        assert inorder_of(tree) == ino
    print("construct_from_preorder_inorder: all tests passed")
