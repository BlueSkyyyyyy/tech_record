"""493. 翻转对（Reverse Pairs）

题目：给定一个数组 nums，如果 i < j 且 nums[i] > 2 * nums[j]，就称 (i, j)
      为一个「重要翻转对」。返回重要翻转对的总数。

思路（分治：归并排序的同时统计跨界的对数）：
    暴力枚举所有 (i, j) 是 O(n^2)。用分治把数组一分为二，任意一对 (i, j)
    按位置关系只可能：
      1. i、j 都在左半；
      2. i、j 都在右半；
      3. i 在左半、j 在右半（跨界）。
    前两类递归统计；第三类在合并阶段单独统计。

    统计跨界对时，一个关键性质是「左右两半都各自有序」——这发生在两半分别
    递归排好序、但还没合并的时候。于是可以：
      对左半每个 i，用一个指针 j 从右半开头往右滑，滑到第一个让
      nums[i] <= 2 * nums[j] 的位置停下；此时右半在 j 之前的元素都满足
      nums[i] > 2 * nums[j]，贡献 j - mid 对。
    因为左半有序，i 增大时 nums[i] 非降、条件更难满足，j 只会继续右移，不会
    回退，所以整个统计是线性的。

    为什么先统计再合并：统计要求左右两半有序才能用单调指针；而合并会破坏
    「分半」的边界，所以统计必须放在 merge 之前。

    为什么用 long long / 长整型比较：2 * nums[j] 可能溢出 32 位 int，
    所以比较时要先转成 64 位再做乘法。

    这与求逆序对（剑指 Offer 51）是同一套骨架，只是比较条件从 nums[i] > nums[j]
    换成了 nums[i] > 2 * nums[j]。

复杂度：时间 O(n log n)（递归 + 每层合并/统计各 O(n)），空间 O(n)（临时数组）。
"""


def reverse_pairs(nums):
    n = len(nums)
    tmp = [0] * n

    def merge_sort(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_sort(lo, mid) + merge_sort(mid, hi)

        j = mid
        for i in range(lo, mid):
            while j < hi and nums[i] > 2 * nums[j]:
                j += 1
            count += j - mid

        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if nums[i] <= nums[j]:
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
            k += 1
        while i < mid:
            tmp[k] = nums[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = nums[j]
            j += 1
            k += 1
        nums[lo:hi] = tmp[lo:hi]
        return count

    return merge_sort(0, n)


if __name__ == "__main__":
    assert reverse_pairs([1, 3, 2, 3, 1]) == 2
    assert reverse_pairs([2, 4, 3, 5, 1]) == 3
    assert reverse_pairs([1, 2, 3, 4]) == 0
    assert reverse_pairs([]) == 0
    assert reverse_pairs([1]) == 0
    assert reverse_pairs([5, 5]) == 0
    assert reverse_pairs([-1, -2]) == 1
    assert reverse_pairs([5, 4, 3, 2, 1]) == 4
    print("reverse_pairs: all tests passed")
