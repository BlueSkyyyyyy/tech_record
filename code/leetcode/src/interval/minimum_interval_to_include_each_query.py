"""1851. 包含每个查询的最小区间（Minimum Interval to Include Each Query）

题目：给定区间列表 intervals[i] = [li, ri] 和查询数组 queries，对每个查询 q，
求包含 q 的区间中长度最短（长度 = ri - li + 1）的那个长度；没有这样的区间返回 -1。

思路（离线排序 + 最小堆，避免重复扫描）：
    如果每个查询都重新扫描所有区间，是 O(nq)。注意「包含 q」的条件是
    `li <= q <= ri`，而 q 越大，能覆盖它的区间集合是单调变化的。于是把查询
    **按 q 从小到大排序**（记下原始下标）后离线处理：
    - 维护一个指针 j，把所有左端点 `li <= 当前 q` 的区间压进一个**最小堆**，
      键是「区间长度」；
    - 堆里可能混入已经过期的区间（右端点 `ri < q`），这时它不可能再覆盖更大的 q，
      直接从堆顶弹掉；
    - 弹完过期项后，堆顶就是覆盖当前 q 的最短区间；堆空则答案是 -1。
    每个区间最多入堆、出堆一次，每个查询一次堆操作。

复杂度：排序 O(n log n + q log q)，处理 O((n + q) log n)；空间 O(n + q)。
"""

import heapq


def min_interval(intervals, queries):
    intervals = sorted(intervals)
    order = sorted(range(len(queries)), key=lambda i: queries[i])
    res = [-1] * len(queries)

    heap = []
    j = 0
    n = len(intervals)
    for i in order:
        q = queries[i]
        while j < n and intervals[j][0] <= q:
            left, right = intervals[j]
            heapq.heappush(heap, (right - left + 1, right))
            j += 1
        while heap and heap[0][1] < q:
            heapq.heappop(heap)
        if heap:
            res[i] = heap[0][0]
    return res


if __name__ == "__main__":
    assert min_interval([[1, 4], [2, 4], [3, 6], [4, 4]], [2, 3, 4, 5]) == [3, 3, 1, 4]
    assert min_interval([[2, 3], [2, 5], [1, 8], [20, 25]], [2, 19, 5, 22]) == [
        2,
        -1,
        4,
        6,
    ]
    assert min_interval([], [1, 2]) == [-1, -1]
    print("minimum_interval_to_include_each_query: all tests passed")
