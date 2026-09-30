"""124. 二叉树中的最大路径和（Binary Tree Maximum Path Sum）

题目：给定一棵二叉树的根节点 root，返回其任意一条路径的最大路径和。
路径被定义为一条从树中任意节点出发、沿父子边移动、到达任意节点的序列，
同一个节点在路径中最多出现一次（路径不必经过根，也不一定经过叶子）。

思路（后序递归：返回「单臂最大增益」，顺路更新全局最优）：
    和 543 直径是同一套骨架，只是把「长度」换成了「和」，并且要考虑负数。

    定义递归函数 `gain(node)`：以 node 为起点、向下的某一条「单臂」路径能贡献的
    最大和（即 node 到它某个后代的路径和）。父节点只能接 left 或 right 其中一条腿，
    所以返回值取两者较大的那个加上自己。

    关键在于负数处理：如果某侧子树的最大增益是负数，那么把这一侧接进来只会让和变小，
    不如不接——所以用 `max(gain, 0)` 把它截断成 0（表示「放弃这一侧」）。
    这样 `node.val + 左增益 + 右增益` 就是「以 node 为最高点的最佳路径」，
    用它更新全局答案；而返回值 `node.val + max(左增益, 右增益)` 只保留单臂继续上传。

    为什么空节点返回 0、且要用 max(..., 0) 截断：
    空子树贡献 0 表示「这里没有节点可选」；
    但如果根节点本身就是负数、且整棵树都在负数里（例如 [-3]），
    用 0 截断会得出「一条都不选」的 0，而题目要求路径必须至少含一个节点，
    所以全局最优的初值要设成负无穷，最终答案取「至少选一个节点」的那个。

复杂度：时间 O(n)（每个节点访问一次），空间 O(h)。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum(root):
    best = float("-inf")

    def gain(node):
        nonlocal best
        if node is None:
            return 0
        left = max(gain(node.left), 0)
        right = max(gain(node.right), 0)
        best = max(best, node.val + left + right)
        return node.val + max(left, right)

    gain(root)
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
    assert max_path_sum(build_tree([1, 2, 3])) == 6
    assert max_path_sum(build_tree([-10, 9, 20, None, None, 15, 7])) == 42
    assert max_path_sum(build_tree([-3])) == -3
    assert max_path_sum(build_tree([2, -1])) == 2
    assert max_path_sum(build_tree([1, -2, -3])) == 1
    assert max_path_sum(build_tree([-2, -1])) == -1
    assert max_path_sum(build_tree([5, 4, 8, 11, None, 13, 4, 7, 2])) == 48
    print("binary_tree_maximum_path_sum: all tests passed")
