"""94. 二叉树的中序遍历（Binary Tree Inorder Traversal）

题目：给定一棵二叉树的根节点 root，返回它的中序遍历（左 → 根 → 右）。

思路（递归遍历框架）：
    中序和先序只差「访问根」这一步的位置：先递归左子树，再记录自己，最后递归右子树。
    其余完全一样——同一个递归框架，调换三行的顺序就得到三种遍历。

    为什么中序特别重要：对**二叉搜索树（BST）**而言，中序遍历的结果一定是
    从小到大的有序序列。这让「验证 BST」「找 BST 第 K 小」「求 BST 中序后继」
    等问题，都能先化归成一次中序遍历。所以记住中序 = 有序这个性质很有用。

    迭代写法的关键动作是「一路向左压栈」：先把从根开始的整条左链压栈，
    弹出栈顶（此刻它没有未访问的左子树）就记录，然后转向它的右子树，重复。
    这个「压左链—弹出—转右」的节奏，正好复刻了递归的调用与返回。

复杂度：时间 O(n)，空间 O(h)（h 为树高）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        result.append(node.val)
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
    assert inorder_traversal(build_tree([])) == []
    assert inorder_traversal(build_tree([1])) == [1]
    assert inorder_traversal(build_tree([1, None, 2, 3])) == [1, 3, 2]
    assert inorder_traversal(build_tree([1, 2, 3, 4, 5])) == [4, 2, 5, 1, 3]
    assert inorder_traversal(build_tree([5, 3, 8, 1, None, 7, 9])) == [1, 3, 5, 7, 8, 9]
    print("inorder_traversal: all tests passed")
