"""80. 删除有序数组中的重复项 II（Remove Duplicates from Sorted Array II）

题目：给你一个非严格递增排列的数组 nums，原地删除重复出现的元素，
使每个元素**最多出现两次**，返回删除后数组的新长度 k。
前 k 个位置应保存最终结果，不需要考虑后面多余的元素。

思路：「快慢指针 + 回头看两位」。
    slow 指向下一个要写入的位置，fast 负责扫描。
    判断 nums[fast] 能否保留的标准是：它与「已写部分的倒数第二个元素」
    nums[slow-2] 是否相同。
      - 若 slow < 2：还没写满两个，任何元素都能保留；
      - 否则若 nums[fast] == nums[slow-2]，说明该值已经出现了至少两次，
        再写就会超过两次，跳过；
    因为数组有序，相同的值总是连续的，所以只需和 nums[slow-2] 比较，
    就能保证每个值最多被写入两次。

复杂度：时间 O(n)，空间 O(1)。
"""


def remove_duplicates_ii(nums):
    slow = 0
    for fast in range(len(nums)):
        if slow < 2 or nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow


if __name__ == "__main__":
    a = [1, 1, 1, 2, 2, 3]
    k = remove_duplicates_ii(a)
    assert k == 5 and a[:k] == [1, 1, 2, 2, 3]

    a = [0, 0, 1, 1, 1, 1, 2, 3, 3]
    k = remove_duplicates_ii(a)
    assert k == 7 and a[:k] == [0, 0, 1, 1, 2, 3, 3]

    a = []
    assert remove_duplicates_ii(a) == 0

    a = [1, 1, 1]
    k = remove_duplicates_ii(a)
    assert k == 2 and a[:k] == [1, 1]
    print("remove_duplicates_sorted_ii: all tests passed")
