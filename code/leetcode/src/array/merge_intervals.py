"""56. 合并区间（Merge Intervals）

题目：给定一个区间数组 intervals，合并所有重叠的区间，返回互不重叠的区间数组。

思路（按左端点排序 + 顺序合并）：
    1. 先按区间的左端点从小到大排序。排序后，能合并的区间一定在数组里相邻，
       不会出现「中间隔着一个已经处理完的区间、后面又蹦出一个能接上的」情况。
    2. 依次扫描：若当前区间的左端点 <= 已合并结果中最后一个区间的右端点，
       说明二者重叠，直接更新那个区间的右端点为两者最大值；否则当前区间
       与前面都不重叠，作为新区间追加。
    为什么排序就能保证正确：排序后左端点单调不减，一旦当前区间与最后一个
    合并区间不重叠（即 start > last_end），后续所有区间的左端点只会更大，
    也不可能再和 last 重叠了，所以可以安全地把它固定下来。

复杂度：排序 O(n log n)，扫描 O(n)；总时间 O(n log n)，空间 O(n)（结果与排序开销）。
"""


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    res = []
    for start, end in intervals:
        if res and start <= res[-1][1]:
            res[-1][1] = max(res[-1][1], end)
        else:
            res.append([start, end])
    return res


if __name__ == "__main__":
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[1, 4], [0, 4]]) == [[0, 4]]
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 2]]) == [[1, 2]]
    print("merge_intervals: all tests passed")
