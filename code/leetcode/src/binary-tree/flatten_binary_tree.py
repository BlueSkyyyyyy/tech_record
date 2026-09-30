"""114. 二叉树展开为链表（Flatten Binary Tree to Linked List）

题目：给定一棵二叉树的根节点 root，把它原地展开成一个「只有右孩子的链表」，
展开后的顺序与二叉树**前序遍历**的顺序一致。要求原地修改，不额外开数组。

思路（后序递归：让每棵子树返回它的尾节点）：
    前序顺序是「根 → 左 → 右」。把左子树整条展开链接到根的右边，再把原来的右子树
    接到左链的尾巴上，就完成了一次「拼接」。对每个节点都这样做，最终整棵树就变成
    前序的右斜链表。

    为了知道「左链的尾巴在哪」，递归函数 `dfs(node)` 返回**以 node 为根的子树
    展开后的最后一个节点（尾节点）**：
      - 空节点返回空；
      - 先递归展开左、右子树，拿到 `left_tail`、`right_tail`；
      - 若左子树存在：把右子树挂到左子树的尾巴后面（`left_tail.right = node.right`），
        再把整条左链搬到右边（`node.right = node.left`），并清空 `node.left`；
      - 返回尾节点：优先 `right_tail`，否则 `left_tail`，都没有就是 `node` 自己。

    为什么用后序：拼接动作依赖「左右子树都已经展开好、且知道各自尾巴」，必须先处理
    子问题再处理当前节点，正是后序。用返回值传尾节点，避免了为找尾巴再遍历一遍。

    另一种等价写法是「逆前序」：按「右 → 左 → 根」的顺序遍历，把每个节点的
    `right` 指向「上一个访问过的节点」，最后一个访问的（原前序第一个）成为新头。
    两种写法都是 O(n) 原地，本篇以「返回尾节点」为主。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)（递归栈）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root):
    def dfs(node):
        if node is None:
            return None
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if node.left is not None:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail or node

    dfs(root)


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


def right_chain(root):
    result = []
    while root is not None:
        result.append(root.val)
        assert root.left is None, "展开后不应再有左孩子"
        root = root.right
    return result


if __name__ == "__main__":
    tree = build_tree([1, 2, 5, 3, 4, None, 6])
    flatten(tree)
    assert right_chain(tree) == [1, 2, 3, 4, 5, 6]

    tree = build_tree([])
    flatten(tree)
    assert right_chain(tree) == []

    tree = build_tree([1])
    flatten(tree)
    assert right_chain(tree) == [1]

    tree = build_tree([1, 2])
    flatten(tree)
    assert right_chain(tree) == [1, 2]

    tree = build_tree([1, None, 2])
    flatten(tree)
    assert right_chain(tree) == [1, 2]

    print("flatten_binary_tree: all tests passed")
