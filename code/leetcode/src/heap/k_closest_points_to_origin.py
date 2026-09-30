"""973. 最接近原点的 K 个点（K Closest Points to Origin）

题目：给定一个点数组 points 和一个整数 k，返回距离原点 (0, 0) 最近的 k 个点。
      两点之间的距离用欧几里得距离；答案顺序任意（可以按距离升序返回）。

思路（维护一个大小为 k 的大顶堆，留下最近的 k 个）：
    这是 215 的「镜像版」：215 求第 K 大，用大小为 k 的**小顶堆**当淘汰线；
    本题求第 K 小（最近），就用大小为 k 的**大顶堆**——堆顶是当前保留集合里
    最远的那个，正好当淘汰线：新点比堆顶近就挤掉堆顶，否则不留。
    遍历完，堆里就是最近的 k 个点。

    为什么比较距离的平方而不是开根号：
    距离是 sqrt(x^2 + y^2)，单调递增，比较大小等价于比较 x^2 + y^2。
    不开根既省时间又避免浮点误差，这是个常用小技巧。

    为什么 Python 里存「负距离的平方」：
    heapq 是小顶堆，要模拟大顶堆只能存相反数；弹出时堆顶对应
    -dist 最小，也就是 dist 最大，即最远的点。

复杂度：时间 O(n log k)，空间 O(k)。
"""

import heapq


def k_closest(points, k):
    max_heap = []
    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(max_heap, (-dist, x, y))
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    return [[x, y] for _, x, y in max_heap]


if __name__ == "__main__":
    assert sorted(k_closest([[1, 3], [-2, 2]], 1)) == [[-2, 2]]
    assert sorted(k_closest([[3, 3], [5, -1], [-2, 4]], 2)) == [[-2, 4], [3, 3]]
    assert sorted(k_closest([[0, 1], [1, 0]], 2)) == [[0, 1], [1, 0]]
    assert sorted(k_closest([[1, 1]], 1)) == [[1, 1]]
    assert sorted(k_closest([[-5, 4], [-6, -5], [4, 6]], 1)) == [[-5, 4]]
    print("k_closest_points_to_origin: all tests passed")
