"""226. 翻转二叉树（Invert Binary Tree）

题目：给定一棵二叉树的根节点 root，翻转它并返回根节点。
「翻转」指把每个节点的左右子树互换。

思路（后序递归，先翻子树再交换）：
    对任意一个节点，翻转后的树 = 左子树翻转后的结果放到右边，
    右子树翻转后的结果放到左边，再交换即可。
    于是递归地：先翻左子树，再翻右子树，最后交换自己的 left 和 right。
    空节点直接返回 None。

    为什么必须先递归再交换：我们要交换的是「两个已经翻转好的子树」。
    如果先交换再递归，同样能对，但概念上稍乱；而「后序处理当前节点」的写法
    与 104、145 一脉相承——当前节点的动作依赖子问题的结果，用后序最清晰。

    为什么不需要返回值也能做：交换本质是原地修改指针，递归函数可以直接改
    节点的 left/right。这里仍返回根节点，只是为了让调用形式统一、方便链式使用。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)（h 为树高）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


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


def preorder(root):
    if root is None:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)


if __name__ == "__main__":
    assert invert_tree(None) is None
    assert preorder(invert_tree(build_tree([1]))) == [1]
    assert preorder(invert_tree(build_tree([4, 2, 7, 1, 3, 6, 9]))) == [4, 7, 9, 6, 2, 3, 1]
    assert preorder(invert_tree(build_tree([1, 2, 3]))) == [1, 3, 2]
    assert preorder(invert_tree(build_tree([1, None, 2]))) == [1, 2]
    print("invert_tree: all tests passed")
