"""437. 路径总和 III（Path Sum III）

题目：给定一棵二叉树的根节点 root 和一个目标和 target_sum，
求树中「方向向下」（从某个祖先到一个后代，可含起点、不含回头）且节点值之和
等于 target_sum 的路径数目。路径不需要从根开始，也不需要在叶子结束。

思路（前缀和 + 哈希表，把数组上的同余技巧搬到树上）：
    在数组里我们做过「和为 K 的子数组」：维护「到当前位置为止的前缀和」，
    用哈希表记录各个前缀和出现过多少次；一段区间和为 target，
    等价于「当前前缀 - 更早的某个前缀 = target」。树上的情形完全一样，
    只不过「前缀」是从根一路走到当前节点的路径和。

    沿用前序遍历，带一个参数 cur = 从根到当前节点的路径和。
    以当前节点为终点、向下的一段路径和恰为 target，当且仅当存在一个祖先节点，
    其前缀和等于 cur - target。所以答案累加 map[cur - target]，
    再把 cur 计数加一，递归左右子树，回溯时把 cur 计数减一。
    初始把空前缀 {0: 1} 放进表里，代表「从根开始」的路径也有一个虚拟前缀。

    为什么必须回溯（把 cur 的计数减回去）：哈希表记录的是「当前这条根到节点
    路径上」出现过的前缀和。一旦离开某个节点去走兄弟分支，它就不再位于当前
    路径上，必须撤销，否则会把不在同一条链上的前缀也配进来，答案偏大。
    这与 113 里 path.pop() 是同一个「做选择 → 递归 → 撤销」的骨架。

    为什么不能用 112/113 那种「到叶子结算」：那里的路径被限定为根到叶子，
    而这里起点、终点都任意。用前缀和，可以一次性统计「以每个节点为终点」的
    所有合法起点，避免了枚举起点再做一次 DFS 的 O(n^2)。

复杂度：时间 O(n)（每个节点访问一次，哈希表操作均摊 O(1)），空间 O(n)（哈希表 + 递归栈）。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def path_sum_iii(root, target_sum):
    prefix_count = {0: 1}

    def dfs(node, cur):
        if node is None:
            return 0
        cur += node.val
        total = prefix_count.get(cur - target_sum, 0)
        prefix_count[cur] = prefix_count.get(cur, 0) + 1
        total += dfs(node.left, cur)
        total += dfs(node.right, cur)
        prefix_count[cur] -= 1
        return total

    return dfs(root, 0)


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
    assert path_sum_iii(build_tree([]), 0) == 0
    assert path_sum_iii(build_tree([5]), 5) == 1
    assert path_sum_iii(build_tree([5]), 4) == 0
    assert path_sum_iii(build_tree([1, 2, 3]), 3) == 2
    assert path_sum_iii(build_tree([1, -2, -3]), -1) == 1
    assert path_sum_iii(build_tree([1, 2, 3, 4, 5]), 3) == 2
    got = path_sum_iii(
        build_tree([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]), 8
    )
    assert got == 3
    print("path_sum_iii: all tests passed")
