"""703. 数据流中的第 K 大元素（Kth Largest Element in a Stream）

题目：设计一个类 KthLargest，初始化时给定 k 和一个初始数组 nums；
      之后每次调用 add(val) 把 val 加入数据流，并返回当前数据流中第 k 大的元素。

思路（维护一个大小为 k 的小顶堆，边来边更新）：
    和 215 是同一个套路，区别在于数据不是一次性给全，而是「持续到来」。
    构造时先把 nums 里的每个数都 add 一遍。
    每次 add(val)：把 val 压入小顶堆，再限制堆大小不超过 k，超过就弹堆顶。
    堆顶就是当前保留的 k 个数里最小的那个，也就是当前数据流的第 k 大。

    为什么这样能在线维护：
    我们永远只需要「最大的 k 个数」，把它们放在一个小顶堆里，
    堆顶就是淘汰线。新数进来只和堆顶比一次：
    比堆顶大就挤掉堆顶、自己留下；比堆顶小就不用留。
    这样无论数据来多少，堆始终只有 k 个元素，内存不随数据总量增长。

    为什么要保留重复元素：题目按「排序后的第 k 个」计数，不去重。
    所以同一个值出现多次，就要在堆里占多个位置，不能当集合处理。

复杂度：构造 O(n log k)；单次 add 时间 O(log k)；空间 O(k)。
"""

import heapq


class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)

    def add(self, val):
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]


if __name__ == "__main__":
    kth = KthLargest(3, [4, 5, 8, 2])
    assert kth.add(3) == 4
    assert kth.add(5) == 5
    assert kth.add(10) == 5
    assert kth.add(9) == 8
    assert kth.add(4) == 8

    kth2 = KthLargest(1, [])
    assert kth2.add(-3) == -3
    assert kth2.add(-2) == -2
    assert kth2.add(-4) == -2
    assert kth2.add(0) == 0
    assert kth2.add(4) == 4

    kth3 = KthLargest(2, [0])
    assert kth3.add(-1) == -1
    assert kth3.add(3) == 0
    assert kth3.add(5) == 3
    print("kth_largest_element_in_a_stream: all tests passed")
