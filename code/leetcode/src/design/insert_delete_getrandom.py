"""380. O(1) 时间插入、删除和获取随机元素（Insert Delete GetRandom O(1)）

题目：实现一个集合，支持在平均 O(1) 时间内：
    insert(val)：插入元素，若已存在返回 False，否则插入并返回 True；
    remove(val)：删除元素，若不存在返回 False，否则删除并返回 True；
    getRandom()：等概率随机返回集合中的一个元素（调用时集合非空）。

思路（动态数组 + 值到下标的反查表）：
    三种能力互相牵制：随机取元素喜欢「数组」（下标均匀），插入删除喜欢「哈希表」
    （O(1) 定位），但数组删除中间元素是 O(n)。破局点在于：数组删除**末尾**是 O(1)，
    而元素的顺序本来就不重要。

    于是用数组 `vals` 存所有元素、哈希表 `pos` 记录「值 → 在 vals 中的下标」：
    - insert：靠 pos 判存在；不存在就追加到末尾，并记下新下标。
    - remove：先查到下标 idx，把**末尾元素**搬到 idx 处填坑（同步更新 pos），
      再 pop 掉末尾。这样只动两个位置，O(1)。注意当删的就是末尾元素时，
      搬运是「自己搬到自己」，先更新 pos 再删也没问题，但次序别写反。
    - getRandom：在 [0, len) 里随机取一个下标，直接返回。

复杂度：insert / remove / getRandom 平均时间 O(1)，空间 O(n)。
"""

import random


class RandomizedSet:
    def __init__(self):
        self.vals = []
        self.pos = {}

    def insert(self, val):
        if val in self.pos:
            return False
        self.pos[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val):
        if val not in self.pos:
            return False
        idx = self.pos[val]
        last = self.vals[-1]
        self.vals[idx] = last
        self.pos[last] = idx
        self.vals.pop()
        del self.pos[val]
        return True

    def getRandom(self):
        return random.choice(self.vals)


if __name__ == "__main__":
    s = RandomizedSet()
    assert s.insert(1) is True
    assert s.remove(2) is False
    assert s.insert(2) is True
    assert s.remove(1) is True
    assert s.insert(2) is False
    assert s.getRandom() == 2
    for v in range(100):
        s.insert(v)
    assert s.remove(37) is True
    assert s.insert(37) is True
    got = [s.getRandom() for _ in range(200)]
    assert all(0 <= x <= 99 or x == 2 for x in got)
    assert len({s.getRandom() for _ in range(2000)}) >= 50
    print("insert_delete_getrandom: all tests passed")
