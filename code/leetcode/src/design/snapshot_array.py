"""1146. 快照数组（Snapshot Array）

题目：实现 SnapshotArray(length)：
    set(index, val)：把下标 index 的值设为 val；
    snap()：拍一次快照，返回快照编号 snap_id（从 0 开始，每次 +1）；
    get(index, snap_id)：返回 index 在编号为 snap_id 的那次快照里的值。
未 set 过的位置默认值为 0。

思路（每个下标只记「变化点」+ 二分查找版本）：
    最直接的做法是每次 snap 都复制整个数组，太浪费。观察发现：一次 set 只改变一个
    下标，其余下标在相邻两次快照之间是重复的。所以为每个下标单独保存一行
    `(快照号, 值)` 的历史，只在值发生变化时追加一条。

    `set` 时如果当前快照号已有记录就覆盖该条，否则追加，避免同一快照号堆多条。
    每行历史的第一条固定是哨兵 `(-1, 0)`，保证任何合法的快照号都能查到结果（默认 0）。
    `get` 时在这一行里二分，找最后一个快照号 <= snap_id 的记录。

    这样空间只与「发生过的修改次数」有关，而不是「快照数 × 数组长度」。

复杂度：set 均摊 O(1)，snap O(1)，get O(log m)（m 为该下标的历史修改次数）；
空间 O(总修改次数 + length)。
"""


class SnapshotArray:
    def __init__(self, length):
        self.snap_id = 0
        self.history = [[[-1, 0]] for _ in range(length)]

    def set(self, index, val):
        row = self.history[index]
        if row[-1][0] == self.snap_id:
            row[-1][1] = val
        else:
            row.append([self.snap_id, val])

    def snap(self):
        sid = self.snap_id
        self.snap_id += 1
        return sid

    def get(self, index, snap_id):
        row = self.history[index]
        lo, hi, ans = 0, len(row) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if row[mid][0] <= snap_id:
                ans = row[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return ans


if __name__ == "__main__":
    arr = SnapshotArray(3)
    arr.set(0, 5)
    assert arr.snap() == 0          # 快照 0：index 0 的值是 5
    arr.set(0, 6)
    assert arr.get(0, 0) == 5       # 快照 0 仍是旧值
    assert arr.get(1, 0) == 0       # 没 set 过，默认 0
    assert arr.snap() == 1          # 快照 1：index 0 的值是 6
    assert arr.get(0, 1) == 6
    arr.set(2, 7)                   # 当前快照号 2 之前先改
    assert arr.snap() == 2
    assert arr.get(2, 2) == 7
    assert arr.get(2, 1) == 0       # 快照 1 时 index 2 还没被改过
    assert arr.get(0, 2) == 6       # index 0 在快照 2 沿用 6
    print("snapshot_array: all tests passed")
