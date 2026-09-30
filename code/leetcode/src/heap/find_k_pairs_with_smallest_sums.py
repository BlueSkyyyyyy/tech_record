"""373. 查找和最小的 K 对数字（Find K Pairs with Smallest Sums）

题目：给定两个以升序排列的整数数组 nums1 和 nums2，以及整数 k。
      定义一对数字 (u, v)：第一个来自 nums1，第二个来自 nums2。
      返回和最小的 k 对数字 (u1, v1), ..., (uk, vk)。

思路（小顶堆 + 单调剪枝，多路归并的又一实例）：
    两个数组都升序，所以对固定的 i，随着 j 增大 nums1[i] + nums2[j] 单调不减。
    于是可以把「每一行 i」看成一个升序序列，问题变成从 k 个有序序列里取最小的 k 个——
    正是 23 题的「多路归并」。
    做法：初始化时，把每个 nums1[i]（i 只取前 k 个就够）与 nums2[0] 组成
    (和, i, 0) 压入小顶堆。每弹出一个 (i, j)，就把它「这一行的下一个」 (i, j+1)
    压回堆。弹出 k 次即得答案。

    为什么 i 只需枚举前 k 个：
    第 j 列同理，答案里的下标不会超过 k-1——
    因为即使最差的情况，也只需要每行/每列贡献少数几个。
    这个剪枝让建堆规模从 O(m) 降到 O(k)。

    为什么堆里要带下标 (i, j) 而不是只存和：
    只存和无法知道它来自哪一行，也就无法取「下一个」。带上下标才方便续推。

复杂度：时间 O(k log k)，空间 O(k)。
"""

import heapq


def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k <= 0:
        return []

    heap = []
    for i in range(min(len(nums1), k)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

    result = []
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return result


if __name__ == "__main__":
    assert k_smallest_pairs([1, 7, 11], [2, 4, 6], 3) == [[1, 2], [1, 4], [1, 6]]
    assert k_smallest_pairs([1, 1, 2], [1, 2, 3], 2) == [[1, 1], [1, 1]]
    assert k_smallest_pairs([1, 2], [3], 3) == [[1, 3], [2, 3]]
    assert k_smallest_pairs([], [1], 1) == []
    assert k_smallest_pairs([1], [], 1) == []
    print("find_k_pairs_with_smallest_sums: all tests passed")
