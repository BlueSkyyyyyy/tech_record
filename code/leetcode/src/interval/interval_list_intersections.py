"""986. 区间列表的交集（Interval List Intersections）

题目：给定两个由**互不相交且已排序**的闭区间组成的列表 firstList、secondList，
返回它们的交集列表。

思路（双指针逐对求交）：
    两个列表各自都有序且互不相交，所以可以用两个指针 i、j 分别扫。每一轮只看
    firstList[i] 与 secondList[j]：
    - 交集为 `[max(l1, l2), min(r1, r2)]`，一个区间存在交集当且仅当左端 <= 右端；
    - 之后把「右端点较小的那个」指针往后移：因为它的右端已经用完了，和后面任何
      区间都不可能再产生新的交集；另一个可能还能和下一对相交。
    这样每个区间至多被访问一次，线性完成。

复杂度：时间 O(m + n)，空间 O(1)（不计返回结果）。
"""


def interval_intersection(first, second):
    res = []
    i = j = 0
    while i < len(first) and j < len(second):
        lo = max(first[i][0], second[j][0])
        hi = min(first[i][1], second[j][1])
        if lo <= hi:
            res.append([lo, hi])
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return res


if __name__ == "__main__":
    first = [[0, 2], [5, 10], [13, 23], [24, 25]]
    second = [[1, 5], [8, 12], [15, 24], [25, 26]]
    assert interval_intersection(first, second) == [
        [1, 2],
        [5, 5],
        [8, 10],
        [15, 23],
        [24, 24],
        [25, 25],
    ]
    assert interval_intersection([[1, 3], [5, 9]], []) == []
    assert interval_intersection([[1, 7]], [[3, 10]]) == [[3, 7]]
    print("interval_intersection: all tests passed")
