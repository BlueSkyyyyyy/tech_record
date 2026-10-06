"""1288. 删除被覆盖区间（Remove Covered Intervals）

题目：给定区间列表 intervals[i] = [li, ri]，删除所有被其他区间**完全覆盖**的区间
（即存在区间 [a, b] 满足 a <= li 且 ri <= b，且不同时为同一个区间），返回剩余区间数。

思路（排序 + 记录最大右端点）：
    先把区间按「左端点升序、左端点相同时右端点降序」排序。这样扫描时，对于当前
    区间 [l, r]，**所有已扫描过的区间左端点都 <= l**（这正是我们需要的一侧条件）。
    于是它被覆盖当且仅当「之前出现过某个右端点 >= r」。只需用一个变量 max_end
    记住扫描过的最大右端点：
    - 若 r > max_end：说明没有区间能盖住它，它保留，答案 +1，并更新 max_end = r；
    - 若 r <= max_end：已有区间的左端更靠左、右端更靠右，它被完全覆盖，跳过。
    左端点相同时把右端从大到小排，能让「更长的那条」先出现，短的自然被覆盖掉。

复杂度：排序 O(n log n)，扫描 O(n)；空间 O(1)（排序本身视语言而定）。
"""


def remove_covered_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    count = 0
    max_end = -1
    for _, end in intervals:
        if end > max_end:
            count += 1
            max_end = end
    return count


if __name__ == "__main__":
    assert remove_covered_intervals([[1, 4], [3, 6], [2, 8]]) == 2
    assert remove_covered_intervals([[1, 4], [2, 3]]) == 1
    assert remove_covered_intervals([[0, 10], [5, 12]]) == 2
    assert remove_covered_intervals([[3, 10], [4, 10], [5, 11]]) == 2
    assert remove_covered_intervals([[1, 2], [1, 4], [3, 4]]) == 1
    print("remove_covered_intervals: all tests passed")
