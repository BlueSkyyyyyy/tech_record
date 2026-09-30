"""347. 前 K 个高频元素（Top K Frequent Elements）

题目：给定一个整数数组 nums 和一个整数 k，返回其中出现频率前 k 高的元素。
      可以按任意顺序返回答案。

思路（先统频率，再用大小为 k 的小顶堆筛出前 k）：
    先用哈希表统计每个元素的出现次数，得到「元素 → 频率」。
    然后遍历这张表，把 (频率, 元素) 压入一个小顶堆，堆顶是频率最小的那对。
    一旦堆的大小超过 k，就弹出堆顶，把频率最低的淘汰掉。
    遍历结束时，堆里剩下的就是频率最高的 k 个元素。

    为什么是「频率入堆」而不是「元素入堆」：
    排序的依据是频率，但输出的是元素，所以把两者打包成一个元组一起进堆，
    让堆按频率自动排序。Python 的元组比较先比频率，频率相同时再比元素，
    这只是一个稳定的兜底规则，不影响正确性。

    为什么又是「小顶堆 + 限制大小 k」：
    和 215 完全同构——要 Top K 大的，就保留一个大小为 k 的小顶堆，
    让堆顶当淘汰线，每次只留最有资格的 k 个。堆里元素数始终是 k，
    和「总共有多少种不同元素」无关，比把所有元素排序快得多。

    另一种做法是桶排序：频率最大不超过 n，用「频率 → 元素列表」的数组，
    从高到低扫桶，取满 k 个即可，时间 O(n)，但需要额外 O(n) 空间。
    这里详展通用、好写的堆解。

复杂度：时间 O(n + m log k)（n 为数组长度，m 为不同元素个数），空间 O(m + k)。
"""

import heapq


def top_k_frequent(nums, k):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    min_heap = []
    for num, freq in count.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for freq, num in min_heap]


if __name__ == "__main__":
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent([1], 1) == [1]
    assert sorted(top_k_frequent([4, 4, 4, 4], 1)) == [4]
    assert sorted(top_k_frequent([1, 2, 1, 2, 1, 2, 3, 3, 3, 3], 1)) == [3]
    print("top_k_frequent_elements: all tests passed")
