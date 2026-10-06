"""2940. 找到 Alice 和 Bob 可以相遇的建筑（Find Building Where Alice and Bob Can Meet）

题目：给定建筑高度数组 heights 和若干查询 [a, b]（a 可大于 b）。两个人从 a、b 出发，
    只能向右移动到「严格更高」的建筑上。求两人能共同到达的最小编号建筑，不能则返回 -1。

思路（线段树 + 在树上二分找第一个更大值）：
    先从规则推出闭式判定。一个人从 i 出发，能到达 j > i 当且仅当 heights[j] > heights[i]：
    因为严格递增链的终点必然比起点高，反之只要 i 与 j 之间满足高度更大就能一步直接跳过去。
    于是若 a == b，答案就是 a。

    否则令 lo = min(a,b)、hi = max(a,b)，答案不可能是 hi 左边（因为 hi 处的两人之一无法左移）。
    检查 j = hi：此时 hi 那个人已就位，只需 lo 那个人能到，即 heights[hi] > heights[lo]，
    成立则答案是 hi。否则需要在 hi 右侧找一个高度严格大于 max(heights[a], heights[b]) 的
    最靠左建筑——这正是「区间第一个大于阈值的位置」。

    用一棵静态最大值线段树：先查右侧区间的最大值是否 > 阈值，若否直接 -1；
    否则从根往下递归，优先走左孩子，第一个满足「子树最大值 > 阈值」的叶子即答案。
    这样每次查询 O(log n)，而不是 O(n) 线性扫描。

复杂度：建树 O(n)，每次查询 O(log n)；空间 O(n)。
"""


def leftmost_building_queries(heights, queries):
    n = len(heights)
    size = 1
    while size < n:
        size *= 2
    tree = [-1] * (2 * size)
    for i, h in enumerate(heights):
        tree[size + i] = h
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[2 * i], tree[2 * i + 1])

    def range_max(l, r):
        res = -1
        l, r = l + size, r + size + 1
        while l < r:
            if l & 1:
                res = max(res, tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, tree[r])
            l //= 2
            r //= 2
        return res

    def first_greater(l, t):
        if l >= n or range_max(l, n - 1) <= t:
            return -1

        def rec(o, nl, nr):
            if nr < l or tree[o] <= t:
                return -1
            if nl == nr:
                return nl
            mid = (nl + nr) // 2
            left = rec(2 * o, nl, mid)
            if left != -1:
                return left
            return rec(2 * o + 1, mid + 1, nr)

        return rec(1, 0, size - 1)

    res = []
    for a, b in queries:
        if a == b:
            res.append(a)
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if heights[hi] > heights[lo]:
            res.append(hi)
        else:
            res.append(first_greater(hi + 1, max(heights[a], heights[b])))
    return res


if __name__ == "__main__":
    assert leftmost_building_queries([6, 4, 8, 5, 2, 7], [[0, 1], [0, 3], [2, 4], [3, 4], [2, 2]]) == [2, 5, -1, 5, 2]
    assert leftmost_building_queries([5, 3, 8, 2, 6, 1, 4, 6], [[0, 7], [3, 5], [5, 2], [3, 0], [1, 6]]) == [7, 6, -1, 4, 6]
    assert leftmost_building_queries([1, 2, 1], [[1, 0]]) == [1]
    print("leftmost_building_queries: all tests passed")
