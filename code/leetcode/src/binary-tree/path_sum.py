"""112. 路径总和（Path Sum）

题目：给定一棵二叉树的根节点 root 和一个目标和 target_sum，
判断树中是否存在「从根节点到叶子节点」的路径，其路径和等于 target_sum。
叶子节点指没有左右孩子的节点。

思路（前序递归，把「还差多少」当参数往下带）：
    对当前节点来说，它只关心「从根走到我之后，还差多少才凑够目标」。
    把 target_sum 减去沿途每个节点的值，到叶子时看剩余量是否恰好为 0。
    这个「剩余量」是由上往下传递的量，天然适合当递归参数带着走——
    进入节点时扣掉自己的值，进入左右子树时把新的剩余量传下去，到叶子结算。

    为什么这里用「传参数」而不是「返回值汇总」：
    路径和是沿着单条链累加出来的，父节点需要把自己的累计值告诉子节点，
    而不是等子节点把某个结果汇总回来。这正是二叉树递归里「前序带着信息
    往下走」的典型形态，与 104 那种「后序把结果往上汇」正好相反。
    选择「携带参数」还是「返回结果」，是树题最核心的分叉口。

复杂度：时间 O(n)（每个节点最多访问一次），空间 O(h)（h 为树高，递归栈深度）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def has_path_sum(root, target_sum):
    if root is None:
        return False
    remaining = target_sum - root.val
    if root.left is None and root.right is None:
        return remaining == 0
    return has_path_sum(root.left, remaining) or has_path_sum(root.right, remaining)


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
    assert has_path_sum(build_tree([]), 0) is False
    assert has_path_sum(build_tree([5]), 5) is True
    assert has_path_sum(build_tree([5]), 4) is False
    assert has_path_sum(build_tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]), 22) is True
    assert has_path_sum(build_tree([1, 2, 3]), 5) is False
    assert has_path_sum(build_tree([1, 2, 3]), 4) is True
    assert has_path_sum(build_tree([-2, None, -3]), -5) is True
    assert has_path_sum(build_tree([1, 2]), 1) is False
    print("path_sum: all tests passed")
