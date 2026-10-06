"""381. O(1) 时间插入、删除和获取随机元素 - 允许重复
     （Insert Delete GetRandom O(1) - Duplicates allowed）

题目：实现 RandomizedCollection，支持平均 O(1)：
    insert(val)：插入 val，返回「插入前集合中是否不存在 val」；
    remove(val)：删除一个 val，返回「是否存在 val 可删」；
    getRandom()：等概率返回集合中的一个元素（按元素个数计，重复元素各算一份）。

思路（动态数组 + 「值 → 下标集合」）：
    与 380（见 19 篇「设计题」）一脉相承，区别是同一个值会出现多次，所以不能再用
    「值 → 单个下标」，而要存「值 → 下标集合」。
    - insert：把值追加到数组末尾，并把新下标记进 pos[val]；返回值就是「pos[val] 的大小
      是否变成 1」。
    - remove：从 pos[val] 任取一个下标 i，用数组末尾元素填坑（数组删末尾是 O(1)）：
        先把 i 从 pos[val] 移除；若 i 不是末尾下标，则把「末尾下标」从 pos[last] 里
        换成 i；再把末尾元素写到 i 处、弹出数组末尾。填坑会动到末尾元素，所以迁移下标
        这一步不能省。
    - getRandom：从数组里等概率取一个下标，返回对应元素。

    注意 val 可能恰好等于末尾元素、或要删的就是末尾元素，这些边界下「迁移下标」要按
    上面次序处理，否则同一集合会被重复写乱。

复杂度：三个操作平均 O(1)；空间 O(n)。
"""

import random


class RandomizedCollection:
    def __init__(self):
        self.nums = []
        self.pos = {}

    def insert(self, val):
        self.nums.append(val)
        self.pos.setdefault(val, set()).add(len(self.nums) - 1)
        return len(self.pos[val]) == 1

    def remove(self, val):
        if val not in self.pos:
            return False
        i = self.pos[val].pop()
        last = self.nums[-1]
        if i != len(self.nums) - 1:
            self.pos[last].discard(len(self.nums) - 1)
            self.pos[last].add(i)
        self.nums[i] = last
        self.nums.pop()
        if not self.pos[val]:
            del self.pos[val]
        return True

    def getRandom(self):
        return random.choice(self.nums)


if __name__ == "__main__":
    random.seed(0)
    c = RandomizedCollection()
    assert c.insert(1) is True
    assert c.insert(1) is False
    assert c.insert(2) is True
    assert c.remove(2) is True
    assert c.remove(2) is False
    assert all(c.getRandom() == 1 for _ in range(200))

    c2 = RandomizedCollection()
    for v in [4, 3, 4, 2, 4]:
        c2.insert(v)
    assert {c2.getRandom() for _ in range(400)} == {2, 3, 4}
    assert c2.remove(4) is True
    assert 4 in {c2.getRandom() for _ in range(400)}
    assert c2.remove(4) and c2.remove(4)
    assert c2.remove(4) is False
    assert {c2.getRandom() for _ in range(200)} == {2, 3}
    print("insert_delete_getrandom_duplicates: all tests passed")
