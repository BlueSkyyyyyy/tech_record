"""525. 连续数组（Contiguous Array）

题目：给定一个只含 0 和 1 的二进制数组 nums，找出含有相同数量 0 和 1 的最长连续子数组，
      并返回该子数组的长度。
      例如 nums = [0, 1, 1, 0, 1, 0]，返回 6（整个数组 0 和 1 各 3 个）。

思路：「0 和 1 数量相等」不好直接用前缀和表达，做一个映射：把 0 看成 -1、1 看成 +1。
      这样一段子数组里 0 和 1 数量相等，当且仅当这段的和为 0。
      于是问题变成「求和为 0 的最长子数组」——正是 560 的特例（k = 0）。
      「和为 0」等价于两个前缀和相等：prefix[i] == prefix[j]。
      所以用哈希表记录每个前缀和「最早」出现的下标；扫描到相同前缀和时就得到一段长度
      i - first[prefix]，用它更新最大值。
      为什么记最早下标：右端固定时，左端越早长度越长，记最早的那个能一次拿到最长长度。
      为什么初始 map 放 {0: -1}：空前缀和为 0、下标 -1，让「从下标 0 开始」的子数组也套同一公式。

复杂度：时间 O(n)，空间 O(n)。
"""


def find_max_length(nums):
    first = {0: -1}
    count = 0
    res = 0
    for i, x in enumerate(nums):
        count += 1 if x == 1 else -1
        if count in first:
            res = max(res, i - first[count])
        else:
            first[count] = i
    return res


if __name__ == "__main__":
    assert find_max_length([0, 1]) == 2
    assert find_max_length([0, 1, 0]) == 2
    assert find_max_length([0, 0, 0, 1, 1, 1]) == 6
    assert find_max_length([1, 1]) == 0
    assert find_max_length([0, 1, 1, 0, 1, 0]) == 6
    print("contiguous_array: all tests passed")
