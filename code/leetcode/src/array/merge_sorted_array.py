"""88. 合并两个有序数组（Merge Sorted Array）

题目：nums1 长度为 m + n，其中前 m 个元素有序；nums2 长度为 n 且有序。
把 nums2 合并进 nums1，使 nums1 整体有序。要求原地完成（nums1 已有足够空间）。

思路：「从后向前的三指针」。
    如果从前往后合并，一旦写入就会覆盖 nums1 里还没处理的小元素，
    除非额外开数组。但 nums1 的尾部是空的（留给合并结果），
    于是换一个方向——都从**后往前**比较，把较大的放到 nums1 的末尾。
    指针 i 指向 nums1 有效部分的末尾，j 指向 nums2 的末尾，
    k 指向 nums1 待写入的末尾 m + n - 1。
    为什么安全：k 始终 >= i，写入位置不会覆盖尚未读取的 nums1 元素；
    且当 j < 0 时 nums2 已取完，剩余 nums1 本来就在正确位置，无需再动。

复杂度：时间 O(m + n)，空间 O(1)。
"""


def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1


if __name__ == "__main__":
    a = [1, 2, 3, 0, 0, 0]
    merge(a, 3, [2, 5, 6], 3)
    assert a == [1, 2, 2, 3, 5, 6]

    a = [1]
    merge(a, 1, [], 0)
    assert a == [1]

    a = [0]
    merge(a, 0, [1], 1)
    assert a == [1]

    a = [4, 5, 6, 0, 0, 0]
    merge(a, 3, [1, 2, 3], 3)
    assert a == [1, 2, 3, 4, 5, 6]
    print("merge_sorted_array: all tests passed")
