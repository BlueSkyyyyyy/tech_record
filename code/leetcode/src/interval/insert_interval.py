"""57. 插入区间（Insert Interval）

题目：给定一个**无重叠、已按左端点升序排列**的区间列表 intervals，以及一个新区间
newInterval，把 newInterval 插入并合并所有重叠部分，返回仍无重叠且有序的结果。

思路（三阶段一次扫描）：
    因为原列表已经有序，插入一个区间只可能影响「中间那一段」。把过程拆成三步：
    1. **新区间左边、完全不相交的区间**：它们的右端点 `intervals[i][1] < start`，
       原样放进结果；
    2. **与新区间重叠的区间**：它们的左端点 `intervals[i][0] <= end`。每遇到一个，
       就用它的两端扩张新区间的 [start, end]（取 min / max），相当于把这段
       连续的重叠块「吸」进新区间；
    3. **新区间右边剩下的区间**：同样完全不相交，原样放进结果。
    最后把扩张后的 [start, end] 放到第 1、3 步结果之间即可。

复杂度：时间 O(n)，空间 O(1)（不计返回结果）。
"""


def insert(intervals, new_interval):
    res = []
    i, n = 0, len(intervals)
    start, end = new_interval

    while i < n and intervals[i][1] < start:
        res.append(intervals[i])
        i += 1

    while i < n and intervals[i][0] <= end:
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])
        i += 1

    res.append([start, end])

    while i < n:
        res.append(intervals[i])
        i += 1

    return res


if __name__ == "__main__":
    assert insert([[1, 3], [6, 9]], [2, 5]) == [[1, 5], [6, 9]]
    assert insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]) == [
        [1, 2],
        [3, 10],
        [12, 16],
    ]
    assert insert([], [5, 7]) == [[5, 7]]
    assert insert([[1, 5]], [2, 3]) == [[1, 5]]
    assert insert([[1, 5]], [6, 8]) == [[1, 5], [6, 8]]
    print("insert_interval: all tests passed")
