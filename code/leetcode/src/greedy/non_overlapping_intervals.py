"""435. 无重叠区间（Non-overlapping Intervals）

题目：给定一个区间的集合 intervals，其中 intervals[i] = [start_i, end_i]。返回需要
移除区间的最小数量，使剩余区间互不重叠。端点相接不算重叠，例如 [1,2] 与 [2,3] 不重叠。

思路（贪心：按右端点排序，优先保留结束最早的区间）：
    要把删掉的区间数降到最少，等价于要保留尽可能多的互不重叠区间。于是问题变成
    「在若干区间里挑出最多的一批，使它们两两不重叠」。
    先把所有区间按右端点从小到大排序，然后从左往右扫，维护「上一个被保留区间的右端点 end」：
      - 若当前区间的左端点 >= end，说明它和已保留的区间不冲突，保留它，更新 end；
      - 否则当前区间与已保留的区间重叠，只能把它删掉（removed++）。
    因为区间已按右端点排序，能保留就保留，且每次保留的都是「结束最早」的那个。

    为什么优先保留结束早的（交换论证）：假设有一个最优方案，它保留的第一个区间是 A，
    而按右端点排序后第一个区间是 B（B 的右端点 <= A 的右端点）。把最优方案里的 A 换成 B：
    B 开始得可能更早，但它的结束更早，B 后面能容纳的区间只会比 A 更多，不会更少。
    所以「每次保留结束最早且不与前面冲突的区间」不会让保留数变少，也不会让删除数变多。

    端点相接不算重叠，所以判断冲突用的是严格小于：当前左端点 < end 才算冲突。

复杂度：时间 O(n log n)（排序），空间 O(1)（排序之外只用常数变量）。
"""


def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    end = intervals[0][1]
    removed = 0
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            removed += 1
        else:
            end = intervals[i][1]
    return removed


if __name__ == "__main__":
    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert erase_overlap_intervals([[1, 2], [1, 2], [1, 2]]) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3]]) == 0
    assert erase_overlap_intervals([]) == 0
    assert erase_overlap_intervals([[1, 100], [11, 22], [1, 11], [2, 12]]) == 2
    print("non_overlapping_intervals: all tests passed")
