"""27. 移除元素（Remove Element）

题目：给定数组 nums 和值 val，原地移除所有等于 val 的元素，
返回新长度 k。前 k 个位置保存结果，顺序不限，后面是什么无所谓。

思路：「快慢指针」。
    slow 指向下一个要写入的位置，fast 负责扫描。
    只要 nums[fast] != val，就把它写到 slow 处并让 slow 前进。
    这等价于把「保留条件」设为「不等于 val」——与 26 题去重是同一套模板。
    用赋值而非交换即可：被跳过的 val 会在后续写入时被覆盖。

复杂度：时间 O(n)，空间 O(1)。
"""


def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow


if __name__ == "__main__":
    a = [3, 2, 2, 3]
    k = remove_element(a, 3)
    assert k == 2 and sorted(a[:k]) == [2, 2]

    a = [0, 1, 2, 2, 3, 0, 4, 2]
    k = remove_element(a, 2)
    assert k == 5 and sorted(a[:k]) == [0, 0, 1, 3, 4]

    a = []
    assert remove_element(a, 1) == 0

    a = [5, 5, 5]
    assert remove_element(a, 5) == 0
    print("remove_element: all tests passed")
