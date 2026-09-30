"""295. 数据流的中位数（Find Median from Data Stream）

题目：设计一个数据结构，支持两种操作：
      addNum(num) 从数据流中添加一个整数；
      findMedian() 返回当前所有元素的中位数。
      偶数个元素时中位数取中间两个数的平均值。

思路（对顶堆：左半边用大顶堆，右半边用小顶堆）：
    把已经读到的数分成两堆：
      left  —— 较小的一半，用大顶堆，堆顶是这半边最大的数；
      right —— 较大的一半，用小顶堆，堆顶是这半边最小的数。
    只要保证两个条件，中位数就唾手可得：
      1) left 的元素个数与 right 相等或恰好比 right 多 1；
      2) left 里每个数都不大于 right 里每个数。
    此时：元素总数为奇数，中位数就是 left 堆顶；
    总数为偶数，中位数就是 (left 堆顶 + right 堆顶) / 2。

    每次 addNum 如何维持这两个条件：
    先把新数压进 left（大顶堆），再把 left 堆顶（左半边最大的）转移到 right，
    相当于给两个堆做了一次「排序归位」，保证条件 2。
    转移后若 right 反而比 left 多，就从 right 把堆顶（右半边最小的）挪回 left，
    保证条件 1。经过这两步，两个不变量始终成立。

    为什么用两个堆而不是每次排序：
    每次取中位数都重新排序是 O(n log n)。对顶堆把插入控制在 O(log n)，
    取中位数 O(1)，非常适合「数据源源不断、随时问中位数」的场景。

    Python 的 heapq 只有小顶堆，所以 left 里存相反数来模拟大顶堆。

复杂度：addNum 时间 O(log n)，findMedian 时间 O(1)，空间 O(n)。
"""

import heapq


class MedianFinder:
    def __init__(self):
        self.left = []   # 大顶堆（存相反数），放较小的一半
        self.right = []  # 小顶堆，放较大的一半

    def add_num(self, num):
        heapq.heappush(self.left, -num)
        heapq.heappush(self.right, -heapq.heappop(self.left))
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))

    def find_median(self):
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2


if __name__ == "__main__":
    finder = MedianFinder()
    finder.add_num(1)
    finder.add_num(2)
    assert finder.find_median() == 1.5
    finder.add_num(3)
    assert finder.find_median() == 2.0

    finder2 = MedianFinder()
    for num in [6, 10, 2, 6, 5, 0, 6, 3, 1, 0, 0]:
        finder2.add_num(num)
    assert abs(finder2.find_median() - 3.0) < 1e-9

    finder3 = MedianFinder()
    finder3.add_num(-1)
    assert finder3.find_median() == -1.0
    finder3.add_num(-2)
    assert finder3.find_median() == -1.5
    print("find_median_from_data_stream: all tests passed")
