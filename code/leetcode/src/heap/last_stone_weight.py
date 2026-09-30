"""1046. 最后一块石头的重量（Last Stone Weight）

题目：有一堆石头，每块石头的重量是正整数。每次选出两块最重的石头，让它们相撞：
      若重量相等，两块都碎掉；否则较重的一块剩下，新重量为两者之差。
      重复到至多剩一块石头，返回它的重量（没有石头则返回 0）。

思路（用大顶堆每次取两个最大值）：
    题目要求「每次取最重的两块」，这正是优先队列最擅长的：
    把所有石头放进一个大顶堆，堆顶就是当前最重的。
    循环：弹出两块最重的 a、b（a >= b），若 a != b，把差值 a - b 压回堆。
    直到堆里不足两块，返回剩下那块的重量（空则 0）。

    为什么用大顶堆而不是每次排序：
    每次模拟都要取当前最大值，若每轮重新排序会退化成 O(n^2 log n)。
    堆能 O(log n) 地取出最大值、O(log n) 地插回新值，
    总共最多做 n 次「取出两块、插回一块」，整体 O(n log n)。

    Python 的 heapq 是小顶堆，所以把石头取负号再放进去；
    取出来时再取负号还原。C++ 的 priority_queue 默认就是大顶堆，直接可用。

复杂度：时间 O(n log n)，空间 O(n)。
"""

import heapq


def last_stone_weight(stones):
    max_heap = [-s for s in stones]
    heapq.heapify(max_heap)
    while len(max_heap) > 1:
        first = -heapq.heappop(max_heap)
        second = -heapq.heappop(max_heap)
        if first != second:
            heapq.heappush(max_heap, -(first - second))
    return -max_heap[0] if max_heap else 0


if __name__ == "__main__":
    assert last_stone_weight([2, 7, 4, 1, 8, 1]) == 1
    assert last_stone_weight([1]) == 1
    assert last_stone_weight([2, 2]) == 0
    assert last_stone_weight([1, 3]) == 2
    assert last_stone_weight([3, 7, 2]) == 2
    assert last_stone_weight([]) == 0
    print("last_stone_weight: all tests passed")
