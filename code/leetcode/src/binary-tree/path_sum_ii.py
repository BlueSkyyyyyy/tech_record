"""113. 路径总和 II（Path Sum II）

题目：给定一棵二叉树的根节点 root 和一个目标和 target_sum，
找出所有「从根节点到叶子节点」且路径和等于 target_sum 的路径。

思路（前序递归携带「剩余和」，并用一个可撤销的当前路径）：
    112 只问「有没有」，113 要「把每一条都找出来」，于是需要一边往下走、
    一边记录当前走过的节点。做法仍是把「还差多少」当参数往下带：
    进入节点时把它的值记进 path、并从剩余和里扣掉；到叶子时若剩余为 0，
    就把 path 的一份快照收进答案。

    关键在回溯：同一层里先走左、再走右，走完左边回到当前节点时，
    path 里不能还残留左边那条路上的节点，否则右边会基于错误的前缀继续拼。
    所以在离开当前节点前要 path.pop()，把状态还原。这正是回溯里
    「做选择 → 递归 → 撤销选择」的标准骨架，只不过选择是「要不要走这个节点」。

    为什么结果要存 path 的副本（Python 的 path.copy()，C++ 的 push_back(path)）：
    path 是全程复用的一个列表，递归返回后会被 pop 修改；若直接把 path 存进去，
    后面所有修改都会串改已收集的答案。必须存一份当时的快照。

复杂度：时间 O(n^2)（每个节点访问一次，最坏如链状树时每收集一条路径要复制 O(n)，
    共 O(n) 条；平衡树下更接近 O(n log n)），空间 O(h)（递归栈 + 当前路径）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def path_sum(root, target_sum):
    result = []
    path = []

    def dfs(node, remaining):
        if node is None:
            return
        path.append(node.val)
        remaining -= node.val
        if node.left is None and node.right is None:
            if remaining == 0:
                result.append(path.copy())
        else:
            dfs(node.left, remaining)
            dfs(node.right, remaining)
        path.pop()

    dfs(root, target_sum)
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
    assert path_sum(build_tree([]), 0) == []
    assert path_sum(build_tree([1, 2]), 1) == []
    assert path_sum(build_tree([1, 2, 3]), 3) == [[1, 2]]

    got = path_sum(build_tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]), 22)
    assert sorted(got) == sorted([[5, 4, 11, 2], [5, 8, 4, 5]])

    got2 = path_sum(build_tree([1, -2, -3, 1, 3, -2, None, -1]), -1)
    assert got2 == [[1, -2, 1, -1]]

    print("path_sum_ii: all tests passed")
