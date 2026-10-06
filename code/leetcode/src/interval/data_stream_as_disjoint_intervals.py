"""352. 将数据流变为多个不相交区间（Data Stream as Disjoint Intervals）

题目：实现一个数据结构，支持两种操作：
    - addNum(val)：往「数据流」里加入一个整数；
    - getIntervals()：返回当前所有数字合并成的不相交区间列表（按左端点升序）。
    重复加入同一个数字不影响结果。

思路（有序列表 + 二分定位，只合并相邻区间）：
    维护一个按左端点升序、互不相交的区间列表 intervals。加入 value 时：
    1. 用二分（bisect_left）找到第一个「左端点 >= value」的位置 i。
       此时 intervals[i-1] 是左边最近的区间、intervals[i] 是右边最近的区间。
    2. **已被覆盖**：若左邻居右端点 >= value，或右邻居左端点 == value，
       什么也不用做。
    3. **与左邻居相邻**：左邻居右端点 == value - 1，把它的右端点扩到 value；
       此时若右邻居左端点 == value + 1，还要把右邻居整段并进来并删掉它。
    4. **只与右邻居相邻**：右邻居左端点 == value + 1，把它的左端点改成 value。
    5. **两边都不挨着**：在位置 i 插入单元素区间 [value, value]。
    每次只可能「吃掉」至多两个邻居，均摊 O(1)，二分 O(log n)。

复杂度：addNum 均摊 O(log n)（二分 + 列表插入 O(n)，n 为区间数）；
        getIntervals O(n)（返回快照）。
"""

import bisect


class SummaryRanges:
    def __init__(self):
        self.intervals = []

    def addNum(self, value):
        intervals = self.intervals
        i = bisect.bisect_left(intervals, [value])

        if i > 0 and intervals[i - 1][1] >= value:
            return
        if i < len(intervals) and intervals[i][0] == value:
            return

        if i > 0 and intervals[i - 1][1] == value - 1:
            intervals[i - 1][1] = value
            if i < len(intervals) and intervals[i][0] == value + 1:
                intervals[i - 1][1] = intervals[i][1]
                intervals.pop(i)
            return

        if i < len(intervals) and intervals[i][0] == value + 1:
            intervals[i][0] = value
            return

        intervals.insert(i, [value, value])

    def getIntervals(self):
        return [list(iv) for iv in self.intervals]


if __name__ == "__main__":
    sr = SummaryRanges()
    sr.addNum(1)
    assert sr.getIntervals() == [[1, 1]]
    sr.addNum(3)
    assert sr.getIntervals() == [[1, 1], [3, 3]]
    sr.addNum(7)
    assert sr.getIntervals() == [[1, 1], [3, 3], [7, 7]]
    sr.addNum(2)
    assert sr.getIntervals() == [[1, 3], [7, 7]]
    sr.addNum(6)
    assert sr.getIntervals() == [[1, 3], [6, 7]]

    sr2 = SummaryRanges()
    sr2.addNum(5)
    sr2.addNum(5)
    assert sr2.getIntervals() == [[5, 5]]
    sr2.addNum(4)
    sr2.addNum(3)
    assert sr2.getIntervals() == [[3, 5]]
    print("summary_ranges_stream: all tests passed")
