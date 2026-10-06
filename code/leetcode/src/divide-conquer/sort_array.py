"""912. 排序数组（Sort an Array，归并排序）

题目：给你一个整数数组 nums，请你将该数组升序排列。

思路（分治：归并排序）：
    分治三步在排序上的直接落地：
      1. 分解：把数组从中间切成左右两半；
      2. 解决：递归地把左半、右半分别排好序；
      3. 合并：把两个「已经有序」的半段合并成一个有序数组。
    合并的诀窍是双指针：两半各拿一个指针指向开头，每次取两个指针里较小的那个
    放进结果，谁被取走谁后移。因为两半各自有序，这样一趟就能合并完。

    为什么这样不会漏掉元素：合并时两半的元素总会被一个指针扫到，且每次只取
    两半当前最小者，保证结果非降序；当一边取完，另一边剩下的直接整体接上即可。

    为什么归并排序值得单独写：它的「分 + 合」结构是分治的教科书模板，很多题
    （求逆序对、翻转对、区间统计）都是在这套骨架上「合的时候多做一点事」。
    这里的合并用「先写进临时数组、再复制回原数组」实现，避免频繁切片。

    另一种写法是自底向上、从长度 1 开始两两合并，不递归；面试里递归版更直观。

复杂度：时间 O(n log n)（T(n) = 2T(n/2) + O(n)，每层合并合计 O(n)，共 log n 层），
    空间 O(n)（临时数组）。
"""


def sort_array(nums):
    n = len(nums)
    tmp = [0] * n

    def merge_sort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        merge_sort(lo, mid)
        merge_sort(mid, hi)

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

    merge_sort(0, n)
    return nums


if __name__ == "__main__":
    assert sort_array([5, 2, 3, 1]) == [1, 2, 3, 5]
    assert sort_array([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
    assert sort_array([]) == []
    assert sort_array([1]) == [1]
    assert sort_array([-3, 7, -3, 0, 7]) == [-3, -3, 0, 7, 7]
    print("sort_array: all tests passed")
