"""268. 丢失的数字（Missing Number）

题目：给定一个包含 0..n 中 n 个数的数组 nums（不重复），找出缺失的那个数。

思路（异或补齐）：
    把「数组里的所有元素」与「本该出现的 0..n」全部异或到一起。两者都出现的数会成对
    抵消（a ^ a = 0），最后剩下的就是只在一侧出现的那个——正是缺失的数。

    也可以理解成：answer = (0 ^ 1 ^ ... ^ n) ^ (nums[0] ^ nums[1] ^ ...)。
    另一种等价做法是求和：0..n 的和减去数组和，但求和可能溢出，异或不会。

复杂度：时间 O(n)，空间 O(1)。
"""


def missing_number(nums):
    result = len(nums)
    for i, x in enumerate(nums):
        result ^= i ^ x
    return result


if __name__ == "__main__":
    assert missing_number([3, 0, 1]) == 2
    assert missing_number([0, 1]) == 2
    assert missing_number([9, 6, 4, 2, 3, 5, 7, 0, 1]) == 8
    assert missing_number([0]) == 1
    assert missing_number([1]) == 0
    print("missing_number: all tests passed")
