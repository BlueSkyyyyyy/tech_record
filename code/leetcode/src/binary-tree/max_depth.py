"""104. 二叉树的最大深度（Maximum Depth of Binary Tree）

题目：给定一棵二叉树的根节点 root，返回它的最大深度。
最大深度是从根节点到最远叶子节点的最长路径上的节点数。

思路（后序递归，用返回值汇总左右子树）：
    以 root 为根的树，最大深度 = 1（根本身）+ 左右子树中更深的那棵的深度。
    这是一个天然的后序递归：先分别问左右子树「你们多深」，等两个答案都回来了，
    再取较大值加一，作为本层的答案往上返回。
    空节点深度为 0，是递归的终止条件。

    为什么用「返回深度」而不是「传参数累计深度」：
    深度这个量是由下往上汇总的——父节点的答案依赖子节点的答案。
    把子树深度作为返回值层层上传，最自然；相反，若把当前深度当参数往下传，
    还要额外维护一个全局最大值去记录见过的最深，反而更绕。
    选择「返回值」还是「参数」，是二叉树递归题的核心设计决策。

复杂度：时间 O(n)（每个节点算一次），空间 O(h)（h 为树高，递归栈深度）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


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
    assert max_depth(build_tree([])) == 0
    assert max_depth(build_tree([1])) == 1
    assert max_depth(build_tree([1, None, 2])) == 2
    assert max_depth(build_tree([3, 9, 20, None, None, 15, 7])) == 3
    assert max_depth(build_tree([1, 2, None, 3, None, 4])) == 4
    print("max_depth: all tests passed")
