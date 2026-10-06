"""384. 打乱数组（Shuffle an Array）

题目：实现 Solution 类：
    Solution(nums)：用数组 nums 初始化；
    reset()：把数组恢复成初始状态并返回；
    shuffle()：等概率随机打乱数组并返回（所有排列出现的概率都相等）。

思路（费雪-耶茨洗牌 / Fisher-Yates）：
    要让 n! 种排列等概率出现，做法是从后往前处理：第 i 位（i 从 n-1 到 1）与
    [0, i] 中的某个位置 j 交换。可以这样理解：先在 n 个位置里等概率挑一个元素放到
    第 n-1 位，再在剩下的 n-1 个里等概率挑一个放到第 n-2 位……每一步的选择都是均匀的，
    所以每个排列的概率都是 1/(n·(n-1)·…·1) = 1/n!。

    必须「从后往前、并与 [0, i]（含自己）交换」。若每次都从整个数组里随机选一个位置
    交换，某些排列会更频繁地出现，分布就不均匀了——这是最经典的写错方式。

复杂度：reset 时间 O(n)、空间 O(n)；shuffle 时间 O(n)、空间 O(1)（不计返回数组）。
"""

import random


class Solution:
    def __init__(self, nums):
        self.original = list(nums)
        self.nums = list(nums)

    def reset(self):
        self.nums = list(self.original)
        return self.nums

    def shuffle(self):
        for i in range(len(self.nums) - 1, 0, -1):
            j = random.randint(0, i)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums


if __name__ == "__main__":
    random.seed(0)
    nums = [1, 2, 3, 4, 5]
    s = Solution(nums)
    seen = set()
    for _ in range(500):
        got = s.shuffle()
        assert sorted(got) == nums
        seen.add(tuple(got))
        assert s.reset() == nums
    assert len(seen) > 100
    print("shuffle_an_array: all tests passed")
