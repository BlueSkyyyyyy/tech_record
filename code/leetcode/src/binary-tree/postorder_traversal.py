"""145. 二叉树的后序遍历（Binary Tree Postorder Traversal）

题目：给定一棵二叉树的根节点 root，返回它的后序遍历（左 → 右 → 根）。

思路（递归遍历框架）：
    后序把「访问根」放到最后：先递归左子树，再递归右子树，最后记录自己。
    三种遍历共用同一个递归框架，只是「根」这一步插在不同位置。

    后序的独特之处在于：递归函数会**在两个子问题都完成之后**才处理当前节点，
    所以它天然适合「需要先知道左右子树的结果，再决定根怎么办」的题目，
    比如求树的高度、判断平衡、计算路径和、翻转二叉树——这些都是后序的形态。

    迭代写法的常用技巧是「按 根→右→左 走一遍，再把结果整体反转」，
    就得到了 左→右→根，即后序。本篇以递归模板为主。

复杂度：时间 O(n)，空间 O(h)（h 为树高）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def postorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        dfs(node.right)
        result.append(node.val)

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
    assert postorder_traversal(build_tree([])) == []
    assert postorder_traversal(build_tree([1])) == [1]
    assert postorder_traversal(build_tree([1, None, 2, 3])) == [3, 2, 1]
    assert postorder_traversal(build_tree([1, 2, 3, 4, 5])) == [4, 5, 2, 3, 1]
    assert postorder_traversal(build_tree([5, 3, 8, 1, None, 7, 9])) == [1, 3, 7, 9, 8, 5]
    print("postorder_traversal: all tests passed")
