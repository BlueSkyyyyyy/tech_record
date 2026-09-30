"""543. 二叉树的直径（Diameter of Binary Tree）

题目：给定一棵二叉树的根节点 root，返回它的直径。
直径指树中任意两个节点之间最长路径的**边数**。

思路（后序递归：返回深度、顺路更新全局最优）：
    一条路径可能不经过根，而是藏在某一棵子树里。因此不能只算「根到最远叶」。
    换个视角：任意一条路径，都可以看成「以某个节点为最高点，向左下走一段、
    向右下走一段」组成。这段路径的长度（边数）正好等于
    **左子树深度 + 右子树深度**。

    于是让递归函数 `depth(node)` 返回「以 node 为根的子树的深度（节点到最远叶的边数）」，
    同时在每个节点处用 `左深 + 右深` 更新答案。父节点拿到的返回值用于继续向上汇总，
    而「经过当前节点的最长路径」只用于更新全局最大值，不再往上返回。

    为什么要区分「返回值」和「全局量」：向上传递时，路径只能走单侧
    （父节点只可能接左边或右边其中一条腿），所以返回值是「单臂长度」；
    而答案允许左右两条腿都算，所以要用一个外部变量记录双臂之和的最大值。
    这是「返回一个量、顺路更新另一个量」的经典题型（同 124 最大路径和）。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameter_of_binary_tree(root):
    best = 0

    def depth(node):
        nonlocal best
        if node is None:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        best = max(best, left + right)
        return 1 + max(left, right)

    depth(root)
    return best


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
    assert diameter_of_binary_tree(build_tree([])) == 0
    assert diameter_of_binary_tree(build_tree([1])) == 0
    assert diameter_of_binary_tree(build_tree([1, 2, 3, 4, 5])) == 3
    assert diameter_of_binary_tree(build_tree([1, 2])) == 1
    assert diameter_of_binary_tree(build_tree([1, 2, None, 3, None, 4, None, 5])) == 4
    print("diameter_of_binary_tree: all tests passed")
