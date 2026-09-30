"""1. 两数之和（Two Sum）

题目：给定一个整数数组 nums 和一个目标值 target，请找出和为 target 的两个整数，
并返回它们的数组下标。每个输入只对应一个答案，且同一个元素不能使用两次。

思路：用哈希表记录「已经扫过的数值 -> 它的下标」。
    遍历到 x 时，先检查表里有没有 target - x：
      - 有：说明前面已经出现过能与 x 配对的数，直接返回两个下标；
      - 没有：把 x 和它的下标存进表里，继续。
    为什么正确：数组无序时无法用对撞双指针一次逼近，但「我在等谁来配对我」这件事
    可以用哈希表记住。每个元素只扫一次，配对的下标关系也不会被覆盖——
    先查后存保证了不会用同一个元素两次。

复杂度：时间 O(n)，空间 O(n)。
"""


def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []


if __name__ == "__main__":
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    assert two_sum([-3, 4, 3, 90], 0) == [0, 2]
    print("two_sum: all tests passed")
