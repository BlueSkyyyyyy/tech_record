"""189. 轮转数组（Rotate Array）

题目：把数组 nums 中的元素向右轮转 k 个位置，要求原地完成，空间尽量省。

思路：「三次反转」。
    直接把每个元素搬到 (i + k) % n 的位置会互相覆盖，需要环状替换并
    小心处理环的起点，代码较绕。三次反转更直观：
      1. 整体反转，得到 [7,6,5,4,3,2,1]；
      2. 反转前 k 个，得到 [5,6,7,4,3,2,1]；
      3. 反转后 n-k 个，得到 [5,6,7,1,2,3,4]。
    为什么对：整体反转把「要移到前面的后缀」和「要后移的前缀」都倒了过来，
    再分别把两段各自反转回来，就恢复了各自内部的顺序，同时交换了两段的位置。
    注意 k 要先对 n 取模，因为轮转 n 次等于没转。

复杂度：时间 O(n)，空间 O(1)。
"""


def rotate(nums, k):
    n = len(nums)
    k %= n

    def reverse(lo, hi):
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo += 1
            hi -= 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7]
    rotate(a, 3)
    assert a == [5, 6, 7, 1, 2, 3, 4]

    a = [-1, -100, 3, 99]
    rotate(a, 2)
    assert a == [3, 99, -1, -100]

    a = [1, 2]
    rotate(a, 0)
    assert a == [1, 2]

    a = [1, 2]
    rotate(a, 4)
    assert a == [1, 2]

    a = [1]
    rotate(a, 1)
    assert a == [1]
    print("rotate_array: all tests passed")
