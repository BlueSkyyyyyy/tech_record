"""968. 监控二叉树（Binary Tree Cameras）

题目：在每个节点上可以安装摄像头，摄像头能覆盖「自己、父节点、直接子节点」。
求覆盖所有节点所需的最少摄像头数量。

思路（树形 DP：后序 + 返回「三状态」）：
    自底向上给每个节点一个状态（用整数表示）：
      0 = 该节点未被覆盖（需要父节点来放摄像头救它）；
      1 = 该节点已被覆盖（自己没有摄像头，但被孩子覆盖了）；
      2 = 该节点装了摄像头。
    后序处理，拿到左右孩子的状态后：
      - 只要有孩子是 0（没被覆盖）：当前节点必须装摄像头 → 状态 2，计数 +1；
      - 否则只要有孩子是 2（装了摄像头）：当前节点就被覆盖 → 状态 1；
      - 否则（两个孩子都是 1，即都「自身被覆盖但没往上传摄像头」）：
        当前节点暂时没被覆盖，交给父节点处理 → 状态 0。
    空节点视为「已被覆盖」（状态 1），避免给叶子误判。
    最后若根节点仍是 0，说明它没人管，必须在根上补一个摄像头。

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


def min_camera_cover(root):
    cameras = 0

    def dfs(node):
        nonlocal cameras
        if node is None:
            return 1
        left = dfs(node.left)
        right = dfs(node.right)
        if left == 0 or right == 0:
            cameras += 1
            return 2
        if left == 2 or right == 2:
            return 1
        return 0

    if dfs(root) == 0:
        cameras += 1
    return cameras


if __name__ == "__main__":
    assert min_camera_cover(build([0, 0, None, 0, 0])) == 1
    assert min_camera_cover(build([0, 0, None, 0, None, 0, None, None, 0])) == 2
    assert min_camera_cover(build([0])) == 1
    assert min_camera_cover(build([0, 0, 0])) == 1
    print("binary_tree_cameras: all tests passed")
