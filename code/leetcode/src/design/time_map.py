"""981. 基于时间的键值存储（Time Based Key-Value Store）

题目：实现 TimeMap：
    set(key, value, timestamp)：存下键 key 在 timestamp 时刻的值；
    get(key, timestamp)：返回 key 在「不晚于 timestamp 的最近一次 set」里的值，
        不存在则返回空串。
同一 key 的 set 时间戳严格递增。

思路（哈希表 + 对时间戳二分）：
    一个 key 会被多次 set，每次带一个递增的时间戳，需要「按时序保留历史、查询时找
    最后一个 <= timestamp 的版本」。这正是「一行历史 + 二分查找」。

    用哈希表把 key 映射到它的两个列表：`times`（递增时间戳）和 `values`（对应值）。
    查询时对 `times` 做 `bisect_right`，得到第一个 > timestamp 的下标再减一，就是
    最后一个 <= timestamp 版本的下标；下标为负说明查询时间早于该 key 的首次 set，
    返回空串。

    因为 set 的时间戳严格递增，追加即可，不需要排序。

复杂度：set 时间 O(1)；get 时间 O(log n)（n 为该 key 的历史次数），空间 O(总 set 次数)。
"""

import bisect


class TimeMap:
    def __init__(self):
        self.times = {}
        self.values = {}

    def set(self, key, value, timestamp):
        self.times.setdefault(key, []).append(timestamp)
        self.values.setdefault(key, []).append(value)

    def get(self, key, timestamp):
        times = self.times.get(key)
        if not times:
            return ""
        i = bisect.bisect_right(times, timestamp) - 1
        return self.values[key][i] if i >= 0 else ""


if __name__ == "__main__":
    tm = TimeMap()
    tm.set("foo", "bar", 1)
    assert tm.get("foo", 1) == "bar"
    assert tm.get("foo", 3) == "bar"        # 3 时刻最近的一版仍是 1 时刻的 bar
    tm.set("foo", "bar2", 4)
    assert tm.get("foo", 4) == "bar2"
    assert tm.get("foo", 5) == "bar2"
    assert tm.get("foo", 3) == "bar"        # 落在两版之间，取更早那版
    assert tm.get("foo", 0) == ""           # 早于首次 set
    assert tm.get("missing", 10) == ""      # 从未出现过
    print("time_map: all tests passed")
