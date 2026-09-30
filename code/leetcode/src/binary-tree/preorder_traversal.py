"""144. 二叉树的前序遍历（Binary Tree Preorder Traversal）

题目：给定一棵二叉树的根节点 root，返回它的前序遍历（根 → 左 → 右）。

思路（递归遍历框架）：
    前序遍历就是「每到一个节点，先记下自己，再递归左子树，最后递归右子树」。
    递归函数只有一个职责：把以自己为根的这棵子树按前序填进结果表。
    终止条件是空节点直接返回。

    为什么递归就够：二叉树本身就是「根 + 左子树 + 右子树」的递归结构，
    遍历的定义天然是递归的。写出「访问根、遍历左、遍历右」这三步，
    树的递归形状会自动把顺序展开成正确的前序序列，不需要额外记录状态。

    迭代写法用显式栈模拟这个顺序：先把根压栈，每次弹出一个就记录，
    再**先压右、后压左**，这样出栈时才是「左先于右」。
    本篇把递归作为主线模板，迭代解在相似题里一句话带过。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)（h 为树高，递归栈深度）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return result


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
    assert preorder_traversal(build_tree([])) == []
    assert preorder_traversal(build_tree([1])) == [1]
    assert preorder_traversal(build_tree([1, None, 2, 3])) == [1, 2, 3]
    assert preorder_traversal(build_tree([1, 2, 3, 4, 5])) == [1, 2, 4, 5, 3]
    assert preorder_traversal(build_tree([1, 2, None, 3, 4])) == [1, 2, 3, 4]
    print("preorder_traversal: all tests passed")
