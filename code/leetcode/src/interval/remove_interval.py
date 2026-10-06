"""1272. 删除区间（Remove Interval）

题目：给定**有序且互不重叠**的区间列表 intervals，以及要删除的区间
toBeRemoved = [lo, hi]，从列表里去掉 [lo, hi] 覆盖的部分，返回剩余区间。

思路（逐个区间判断「被切掉哪一块」）：
    对每个区间 [a, b] 与删除区间 [lo, hi]：
    - **完全不相交**（`b <= lo` 或 `a >= hi`）：原样保留；
    - **相交**：它可能只被切左端、只被切右端，或（删除区间完全落在它内部时）
      被切成左右两段。统一写成：
      - 若 `a < lo`，保留左残段 [a, lo]；
      - 若 `b > hi`，保留右残段 [hi, b]。
      当 [lo, hi] 完全覆盖 [a, b] 时两个条件都不成立，什么都不留，正好对应删除。
    因为原列表有序，残段自然仍有序。

复杂度：时间 O(n)，空间 O(1)（不计返回结果）。
"""


def remove_interval(intervals, to_be_removed):
    lo, hi = to_be_removed
    res = []
    for a, b in intervals:
        if b <= lo or a >= hi:
            res.append([a, b])
        else:
            if a < lo:
                res.append([a, lo])
            if b > hi:
                res.append([hi, b])
    return res


if __name__ == "__main__":
    assert remove_interval([[0, 2], [3, 4], [5, 7]], [1, 6]) == [[0, 1], [6, 7]]
    assert remove_interval([[0, 5]], [1, 3]) == [[0, 1], [3, 5]]
    assert remove_interval([[0, 5]], [-5, -1]) == [[0, 5]]
    assert remove_interval([[0, 2], [3, 4], [5, 7]], [-5, -1]) == [
        [0, 2],
        [3, 4],
        [5, 7],
    ]
    assert remove_interval([[-5, -4], [-3, -2], [-1, 0]], [-3, -2]) == [
        [-5, -4],
        [-1, 0],
    ]
    print("remove_interval: all tests passed")
