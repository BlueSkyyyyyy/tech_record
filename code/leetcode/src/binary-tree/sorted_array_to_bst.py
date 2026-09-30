"""108. 将有序数组转换为二叉搜索树（Convert Sorted Array to BST）

题目：给定一个按升序排列的整数数组 nums，将它转换成一棵**高度平衡**的二叉搜索树，
返回它的根节点。（本题有多个合法答案。）

思路（分治：取中点作根，左右递归）：
    要让 BST 高度平衡，最自然的做法是让左右子树的节点数尽量相等。
    有序数组的中点正好把数组分成「比它小的左半」和「比它大的右半」，
    于是取中点作根，再对左右两半分别递归，就同时满足了：
      - 左半全小于根、右半全大于根 → 是 BST；
      - 左右规模相差不超过 1 → 高度平衡。

    这正是「中序遍历序列反推 BST」的过程：中序有序，所以只要保证每次取的是
    当前区间的中间元素作根，还原出的树就平衡。用下标区间 [lo, hi] 递归，
    不切数组，每个元素只被处理一次。

    为什么取中点而不是首尾：取首或尾会让树退化成链表（高度 O(n)）；
    取中点使两边规模均衡，高度稳定在 O(log n)。题目若要求「任意一种合法答案」，
    左右半边取哪个中点都可以，取 `(lo + hi) // 2` 是常见选择。

复杂度：时间 O(n)（每个元素建一个节点），空间 O(log n)（平衡树的递归栈深度）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_array_to_bst(nums):
    def build(lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = TreeNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(nums) - 1)


def inorder(node, out):
    if node is None:
        return
    inorder(node.left, out)
    out.append(node.val)
    inorder(node.right, out)


def height(node):
    if node is None:
        return 0
    return 1 + max(height(node.left), height(node.right))


if __name__ == "__main__":
    for nums in ([], [-10], [1, 3], [1, 2, 3, 4, 5], [-10, -3, 0, 5, 9]):
        tree = sorted_array_to_bst(nums)
        out = []
        inorder(tree, out)
        assert out == nums, "中序必须还原出有序数组"
        left_h = height(tree.left) if tree else 0
        right_h = height(tree.right) if tree else 0
        assert abs(left_h - right_h) <= 1, "必须高度平衡"
    print("sorted_array_to_bst: all tests passed")
