"""26. 删除有序数组中的重复项（Remove Duplicates from Sorted Array）

题目：给你一个非严格递增排列的数组 nums，原地删除重复出现的元素，
使每个元素只出现一次，返回删除后数组的新长度 k。前 k 个位置应保存最终结果，
不需要考虑后面多余的元素。

思路：「快慢指针」。
    slow 指向已去重部分的最后一个位置，fast 扫描整个数组。
    由于数组有序，重复元素必然相邻：只要 nums[fast] != nums[slow]，
    就说明遇到了一个新值，把它写到 slow 的下一位。
    为什么有序是关键：无序时无法用一次线性扫描判断「是否出现过」。

复杂度：时间 O(n)，空间 O(1)。
"""


def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1


if __name__ == "__main__":
    a = [1, 1, 2]
    k = remove_duplicates(a)
    assert k == 2 and a[:k] == [1, 2]

    a = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    k = remove_duplicates(a)
    assert k == 5 and a[:k] == [0, 1, 2, 3, 4]

    a = []
    assert remove_duplicates(a) == 0

    a = [7]
    assert remove_duplicates(a) == 1 and a[:1] == [7]
    print("remove_duplicates_sorted: all tests passed")
