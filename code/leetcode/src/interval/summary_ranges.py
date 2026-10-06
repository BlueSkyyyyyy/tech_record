"""228. 汇总区间（Summary Ranges）

题目：给定一个**无重复元素的有序整数数组** nums，返回恰好覆盖数组中所有数字的
最小有序区间范围列表。每个区间写成 "a"（单个数字）或 "a->b"（连续的一段）。

思路（一次扫描，找连续段）：
    数组已经有序且无重复，所以「连续」就是 `nums[i + 1] == nums[i] + 1`。
    - 用下标 i 指向当前段的起点 start；
    - 只要下一个数正好比当前数大 1，就把 i 往后挪；
    - 挪不动了，[start, nums[i]] 就是一段：两端相等输出 "start"，
      否则输出 "start->nums[i]"；
    - 再把 i 前进一步，开始找下一段。
    全程只扫一遍数组。

复杂度：时间 O(n)，空间 O(1)（不计返回结果）。
"""


def summary_ranges(nums):
    res = []
    i, n = 0, len(nums)
    while i < n:
        start = nums[i]
        while i + 1 < n and nums[i + 1] == nums[i] + 1:
            i += 1
        if start == nums[i]:
            res.append(str(start))
        else:
            res.append(f"{start}->{nums[i]}")
        i += 1
    return res


if __name__ == "__main__":
    assert summary_ranges([0, 1, 2, 4, 5, 7]) == ["0->2", "4->5", "7"]
    assert summary_ranges([0, 2, 3, 4, 6, 8, 9]) == ["0", "2->4", "6", "8->9"]
    assert summary_ranges([]) == []
    assert summary_ranges([-1]) == ["-1"]
    print("summary_ranges: all tests passed")
