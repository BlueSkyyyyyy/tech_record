"""398. 随机数索引（Random Pick Index）

题目：给定可能含有重复元素的数组 nums，实现 pick(target)：等概率返回一个满足
    nums[i] == target 的下标 i。

思路（水塘抽样）：
    只关心等于 target 的位置。一次遍历：设当前位置是第 count 个等于 target 的数，
    以 1/count 的概率把答案换成它的下标，否则保持。道理与 382 相同——
    每个目标下标最终被留下的概率都是 1/(target 出现的次数)，所以是等概率。
    这样无需先把所有下标存下来，额外空间 O(1)。

复杂度：初始化 O(1)；pick 时间 O(n)、空间 O(1)。
"""

import random


class Solution:
    def __init__(self, nums):
        self.nums = nums

    def pick(self, target):
        res = -1
        count = 0
        for i, x in enumerate(self.nums):
            if x == target:
                count += 1
                if random.randint(1, count) == 1:
                    res = i
        return res


if __name__ == "__main__":
    random.seed(0)
    s = Solution([1, 2, 3, 3, 3, 2])
    seen = set()
    for _ in range(4000):
        i = s.pick(3)
        assert s.nums[i] == 3
        seen.add(i)
    assert seen == {2, 3, 4}
    assert {s.pick(2) for _ in range(500)} == {1, 5}
    print("random_pick_index: all tests passed")
