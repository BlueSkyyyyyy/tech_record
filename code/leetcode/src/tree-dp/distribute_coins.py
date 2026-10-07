"""979. 在二叉树中分配硬币（Distribute Coins in Binary Tree）

题目：树上有 n 个节点，每个节点有硬币若干，总硬币数恰好等于节点数。
每次可以把一枚硬币从节点移到相邻节点。求让每个节点恰好有一枚硬币的最少移动次数。

思路（树形 DP：后序 + 返回「净流量」）：
    把每棵子树看成一个整体：它最终必须有「节点数」枚硬币（因为每节点一枚）。
    递归返回该子树「多出来 / 缺少」的硬币数 `balance = 子树硬币数 - 子树节点数`：
      - balance > 0：多余，需要向外送去 balance 枚；
      - balance < 0：缺少，需要从外部运进 -balance 枚。
    跨越「当前节点与某个孩子」这条边的硬币数，恰好等于该孩子子树的 |balance|——
    多出来的要运出去，缺的要运进来，无论方向，都要走这条边 |balance| 次。
    把所有边的 |balance| 累加即为答案。

复杂度：时间 O(n)，空间 O(h)。
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(vals):
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root


def distribute_coins(root):
    moves = 0

    def dfs(node):
        nonlocal moves
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        moves += abs(left) + abs(right)
        return node.val + left + right - 1

    dfs(root)
    return moves


if __name__ == "__main__":
    assert distribute_coins(build([3, 0, 0])) == 2
    assert distribute_coins(build([0, 3, 0])) == 3
    assert distribute_coins(build([1])) == 0
    assert distribute_coins(build([1, 0, 2])) == 2
    print("distribute_coins: all tests passed")
