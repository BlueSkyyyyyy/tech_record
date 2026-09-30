"""102. 二叉树的层序遍历（Binary Tree Level Order Traversal）

题目：给定一棵二叉树的根节点 root，返回它的层序遍历结果。
即逐层地、从左到右访问所有节点，结果是一个「列表的列表」。

思路（BFS + 队列，逐层切分）：
    层序遍历用队列：先把根入队，然后不断「出队一个节点、把它的左右孩子入队」，
    出队顺序自然就是逐层从左到右。难点在于如何把结果**按层分组**。

    技巧是每次进入 while 循环时先记下 `len(queue)`，这就是当前层的节点数；
    然后只出队这么多个节点，作为本层结果收集起来，新入队的孩子属于下一层。
    循环一次处理完一整层，把这一层加入结果即可。空树返回空列表。

    为什么不能只用一个队列直接铺平输出：题目要的是分层结构（二维）。
    先数当前层大小再定长出队，是用 BFS 天然分层的最省事写法；
    也可以在处理每个节点时往结果里塞一个「层号」，但那样要多带状态。

复杂度：时间 O(n)（每个节点进出队一次），空间 O(w)（w 为树的最大宽度）。
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def level_order(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        result.append(level)
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
    assert level_order(build_tree([])) == []
    assert level_order(build_tree([1])) == [[1]]
    assert level_order(build_tree([3, 9, 20, None, None, 15, 7])) == [
        [3],
        [9, 20],
        [15, 7],
    ]
    assert level_order(build_tree([1, 2, 3, None, 4, None, 5])) == [
        [1],
        [2, 3],
        [4, 5],
    ]
    print("level_order: all tests passed")
