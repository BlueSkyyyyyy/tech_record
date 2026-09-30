"""283. 移动零（Move Zeroes）

题目：给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，
同时保持非零元素的相对顺序。必须原地操作，不能额外复制数组。

思路：「快慢指针」的原地覆盖。
    slow 指向下一个「非零元素应放的位置」，fast 遍历数组：
      - nums[fast] != 0：把它交换到 slow 处，slow 右移；
      - nums[fast] == 0：跳过。
    交换（而非直接赋值）保证零自然被换到后面；用赋值也能做（最后补零），
    但交换写法更直观，且只遍历一遍。

复杂度：时间 O(n)，空间 O(1)。
"""


def move_zeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


if __name__ == "__main__":
    a = [0, 1, 0, 3, 12]
    move_zeroes(a)
    assert a == [1, 3, 12, 0, 0]

    a = [0]
    move_zeroes(a)
    assert a == [0]

    a = [1, 2, 3]
    move_zeroes(a)
    assert a == [1, 2, 3]

    a = [0, 0, 0, 1]
    move_zeroes(a)
    assert a == [1, 0, 0, 0]
    print("move_zeroes: all tests passed")
