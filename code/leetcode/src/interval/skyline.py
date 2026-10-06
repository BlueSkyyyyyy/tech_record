"""218. 天际线问题（The Skyline Problem）

题目：给定一组矩形的左 x、右 x、高度 [left, right, height]，返回这些建筑构成的
天际线「关键点」列表（每个点 [x, y] 表示在横坐标 x 处天际线高度变为 y）。

思路（扫描线 + 记录最大高度的大顶堆）：
    把每栋楼拆成两个事件：在 left 处「进入」（记录高度），在 right 处「离开」。
    把所有事件按 x 升序排序；同一 x 上让「进入」排在「离开」前面（这样不会
    在新楼还没加进来之前就误报高度下降）。
    从左到右扫事件，用一个**大顶堆**维护当前仍然覆盖着的楼高（堆里同时存右端点，
    用于惰性删除）：
    - 遇到进入事件就把高度入堆；
    - 把堆顶那些「右端点 <= 当前 x」的楼层弹出（它们已经离开了）；
    - 堆顶高度就是当前 x 之后的天际线高度；若与上一个记录的高度不同，就产生一个
      关键点 [x, 高度]。
    地面高度用 0 兜底（预先放一个不会过期的 0）。

复杂度：事件排序 O(n log n)，每个事件 O(log n)；空间 O(n)。
"""

import heapq


def get_skyline(buildings):
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))  # 进入：高度取负，便于统一排序
        events.append((right, height, right))  # 离开：高度为正
    events.sort()

    res = []
    heap = [(0, float("inf"))]  # (-高度, 右端点)，地面常驻
    for x, signed_height, right in events:
        if signed_height < 0:
            heapq.heappush(heap, (signed_height, right))
        while heap[0][1] <= x:
            heapq.heappop(heap)
        cur = -heap[0][0]
        if not res or res[-1][1] != cur:
            res.append([x, cur])
    return res


if __name__ == "__main__":
    assert get_skyline(
        [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
    ) == [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]]
    assert get_skyline([[0, 2, 3], [2, 5, 3]]) == [[0, 3], [5, 0]]
    assert get_skyline([[1, 2, 1], [1, 2, 2], [1, 2, 3]]) == [[1, 3], [2, 0]]
    print("skyline: all tests passed")
