"""15. 三数之和（3Sum）

题目：给你一个整数数组 nums，判断是否存在三元组 [nums[i], nums[j], nums[k]]
满足 i != j、i != k、j != k，且 nums[i] + nums[j] + nums[k] == 0。
返回所有和为 0 且不重复的三元组。

思路：「排序 + 固定一个数 + 对撞双指针」。
    1. 先排序，这样可以用双指针，也让重复元素相邻、便于去重；
    2. 外层枚举第一个数 nums[i]，内层在 (i, n-1] 上用对撞双指针找两数之和 = -nums[i]；
    3. 去重是本题最容易错的地方，要处理两处：
       - 外层：若 nums[i] == nums[i-1]，跳过，避免同一首数重复；
       - 找到一组解后，lo/hi 各自跳过相邻的相同值，再继续收缩。
    剪枝：排序后若 nums[i] > 0，则后面都是正数，和不可能为 0，直接 break。

复杂度：排序 O(n log n)，双指针主体 O(n^2)，总时间 O(n^2)，空间 O(1)（不计排序与输出）。
"""


def three_sum(nums):
    nums = sorted(nums)
    n = len(nums)
    res = []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
    return res


if __name__ == "__main__":
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([]) == []
    assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([-2, 0, 0, 2, 2]) == [[-2, 0, 2]]
    print("three_sum: all tests passed")
