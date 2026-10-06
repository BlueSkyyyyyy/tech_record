"""53. 最大子数组和（Maximum Subarray，分治解）

题目：给你一个整数数组 nums，请你找出一个具有最大和的连续子数组
（子数组最少包含一个元素），返回其最大和。

思路（分治：最近子数组要么在左半、要么在右半、要么横跨中点）：
    把数组从中间一切为二，任意一段连续子数组和「分界点」只有三种关系：
      1. 完全落在左半段；
      2. 完全落在右半段；
      3. 横跨分界点（左半占一截、右半占一截）。
    前两种情况递归求解即可；第三种情况需要单独处理。

    横跨中点的子数组，一定是从 mid 往左延伸的一段连续后缀 + 从 mid 往右延伸的
    一段连续前缀。为了让总和最大，左右两侧各自都取「从 mid 出发的最大和」：
    分别向左、向右累加并记录过程中出现过的最大累加值，两者相加就是跨越中点的
    最大和。最后三种情况取最大，就是整段的答案。

    为什么这样不会漏解：任何连续子数组，或者不跨 mid（归入前两类），或者跨 mid
    （归入第三类）。三类穷尽且互不遗漏，所以归并出来的就是全局最优。

    另一种更出名的解法是动态规划（Kadane）：dp[i] 表示「以 i 结尾的最大子数组和」，
    一次遍历 O(n) 即可，见动态规划篇。分治解较慢，但它是「合并层需要额外计算」
    这一类分治题的典型模板，与归并排序求逆序对、翻转对同源。

复杂度：时间 O(n log n)（T(n) = 2T(n/2) + O(n)，每层合并扫描 O(n)），
    空间 O(log n)（递归栈深度）。
"""


def max_subarray(nums):
    def solve(lo, hi):
        if hi - lo == 1:
            return nums[lo]
        mid = (lo + hi) // 2
        left_best = solve(lo, mid)
        right_best = solve(mid, hi)

        best = float("-inf")
        total = 0
        for i in range(mid - 1, lo - 1, -1):
            total += nums[i]
            best = max(best, total)
        left_cross = best

        best = float("-inf")
        total = 0
        for i in range(mid, hi):
            total += nums[i]
            best = max(best, total)
        right_cross = best

        return max(left_best, right_best, left_cross + right_cross)

    return solve(0, len(nums))


if __name__ == "__main__":
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray([1]) == 1
    assert max_subarray([5, 4, -1, 7, 8]) == 23
    assert max_subarray([-3, -1, -2]) == -1
    assert max_subarray([-1, -2]) == -1
    print("maximum_subarray: all tests passed")
