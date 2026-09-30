"""215. 数组中的第 K 个最大元素（Kth Largest Element in an Array）

题目：给定整数数组 nums 和整数 k，返回数组中第 k 个最大的元素。
      注意是「排序后的第 k 个最大」，而不是第 k 个不同的元素。

思路（维护一个大小为 k 的小顶堆）：
    遍历数组，把每个数压入一个小顶堆（堆顶最小）。
    一旦堆的大小超过 k，就弹出堆顶——被弹掉的永远是「当前已见过的数里最小的那个」。
    遍历结束时，堆里留下的是整个数组中最大的 k 个数，而堆顶就是这 k 个里最小的，
    也就是全局第 k 大。

    为什么用「小顶堆 + 限制大小 k」而不是「大顶堆」：
    我们要的是第 k 大，相当于只要保留最大的 k 个就够了，多余的小数直接扔掉。
    小顶堆的堆顶天生就是「当前保留集合里最小的」，正好是淘汰线：
    新来一个数只要比堆顶大，就值得挤进来，把堆顶踢掉；比堆顶小则连进都不用进。
    这样堆里始终只有 k 个元素，时间和空间都只与 k 有关，而不是与 n 有关。

    另一种做法是快速选择（quickselect），平均 O(n)，但最坏 O(n^2)、实现也更容易写错，
    这里只详展最稳妥的堆解。

复杂度：时间 O(n log k)（每个元素入堆一次，堆高为 log k），空间 O(k)。
"""

import heapq


def find_kth_largest(nums, k):
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]


if __name__ == "__main__":
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert find_kth_largest([1], 1) == 1
    assert find_kth_largest([-1, -2, -3], 3) == -3
    assert find_kth_largest([7, 7, 7], 2) == 7
    print("kth_largest_element_in_an_array: all tests passed")
