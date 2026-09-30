"""349. 两个数组的交集（Intersection of Two Arrays）

题目：给定两个数组 nums1 和 nums2，返回它们的交集。
结果中的每个元素必须唯一，顺序不限。

思路：题目只要求「元素是否出现」而不关心次数，所以先把 nums1 装进哈希集合。
    再遍历 nums2，遇到出现在集合里的数就收集，同时从集合里删掉，
    这样同一个数不会被收进结果两次，天然满足「元素唯一」。
    若改成保留出现次数（350 题），就要换成计数字典/多重集。

复杂度：时间 O(m + n)，空间 O(m)（或 O(min(m, n))）。
"""


def intersection(nums1, nums2):
    set1 = set(nums1)
    res = []
    for x in nums2:
        if x in set1:
            res.append(x)
            set1.discard(x)
    return res


if __name__ == "__main__":
    assert sorted(intersection([1, 2, 2, 1], [2, 2])) == [2]
    assert sorted(intersection([4, 9, 5], [9, 4, 9, 8, 4])) == [4, 9]
    assert intersection([1, 2, 3], [4, 5, 6]) == []
    assert sorted(intersection([1, 1, 1], [1, 1])) == [1]
    print("intersection_of_two_arrays: all tests passed")
