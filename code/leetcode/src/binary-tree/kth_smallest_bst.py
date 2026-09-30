"""230. 二叉搜索树中第 K 小的元素（Kth Smallest Element in a BST）

题目：给定一棵二叉搜索树的根节点 root 和一个整数 k，返回其中第 k 小的元素
（k 从 1 开始计数）。

思路（中序遍历 + 计数，边走边停）：
    二叉搜索树有一条最重要的性质：**中序遍历的结果是严格递增的**。
    所以「第 k 小」就等于「中序遍历序列里的第 k 个元素」。

    于是只要做一次中序遍历，用一个计数器记录「已经访问到第几个」，
    数到第 k 个时直接把它作为答案返回。

    为什么中序天然有序：中序的访问顺序是「左子树 → 根 → 右子树」，
    而 BST 保证左子树的值全小于根、右子树的值全大于根。
    递归展开后，任意一个节点都在它所有右子树节点之前、所有左子树节点之后，
    整体就是升序。

    为什么可以提前停：一旦数到第 k 个，后面更大的元素都不可能是答案，
    直接返回即可，不必遍历完整棵树。最好情况下（k 很小）能省不少时间。

复杂度：时间 O(h + k)（从根走到最左下角 O(h)，再访问 k 个节点），
        最坏 O(n)；空间 O(h)（递归栈）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kth_smallest(root, k):
    count = 0
    answer = None

    def inorder(node):
        nonlocal count, answer
        if node is None or answer is not None:
            return
        inorder(node.left)
        count += 1
        if count == k:
            answer = node.val
            return
        inorder(node.right)

    inorder(root)
    return answer


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
    assert kth_smallest(build_tree([3, 1, 4, None, 2]), 1) == 1
    assert kth_smallest(build_tree([3, 1, 4, None, 2]), 2) == 2
    assert kth_smallest(build_tree([3, 1, 4, None, 2]), 3) == 3
    assert kth_smallest(build_tree([3, 1, 4, None, 2]), 4) == 4
    assert kth_smallest(build_tree([5, 3, 6, 2, 4, None, None, 1]), 3) == 3
    assert kth_smallest(build_tree([5, 3, 6, 2, 4, None, None, 1]), 6) == 6
    print("kth_smallest_bst: all tests passed")
