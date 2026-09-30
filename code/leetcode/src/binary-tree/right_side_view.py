"""199. 二叉树的右视图（Binary Tree Right Side View）

题目：给定一棵二叉树的根节点 root，想象自己站在它的右侧，
按从顶部到底部的顺序，返回从右侧能看到的节点值。

思路（层序遍历，取每层最后一个）：
    「从右侧看得到」的节点，正是每一层里最靠右的那个。
    于是先做一次 102 的层序遍历，每一层只保留最后一个节点即可。
    在按层出队时判断「是不是本层最后一个」（下标等于本层大小减一），
    是的就收集它的值。

    为什么用 BFS 而不是 DFS：右视图按「层」定义，BFS 天然按层推进，
    取每层末元素就是答案，逻辑最直白。DFS 也能做，比如
    「先访问右孩子、每个深度只记录第一次遇到的节点」，但要额外维护深度集合，
    不如 BFS 直观。

复杂度：时间 O(n)（每个节点进出队一次），空间 O(w)（树的最大宽度）。
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def right_side_view(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                result.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
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
    assert right_side_view(build_tree([])) == []
    assert right_side_view(build_tree([1])) == [1]
    assert right_side_view(build_tree([1, 2, 3, None, 5, None, 4])) == [1, 3, 4]
    assert right_side_view(build_tree([1, 2, 3, 4])) == [1, 3, 4]
    assert right_side_view(build_tree([1, None, 2, 3])) == [1, 2, 3]
    print("right_side_view: all tests passed")
