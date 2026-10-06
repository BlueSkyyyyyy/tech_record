"""706. 设计哈希映射（Design HashMap）

题目：不使用任何内建的哈希表库，设计一个哈希映射。支持三个操作：
    put(key, value)：插入或更新键值对；
    get(key)：按键取值，不存在返回 -1；
    remove(key)：按键删除，不存在则什么都不做。

思路（定长桶数组 + 链地址法）：
    哈希表的两件事是「把键映射到一个位置」和「冲突了怎么办」。
    这里用一个长度为 `size` 的桶数组，位置 = key % size；每个桶挂一条链
    （Python 用 list，C++ 用 vector），把落在同一个位置的键值对串起来。
    - put：先在该桶里找键，找到就改值，否则追加；
    - get：在该桶里找键，找到返回值，否则 -1；
    - remove：在该桶里删除该键。

    为什么不直接用大小为 10^6+1 的数组：那当然也是 O(1)，但等于把「哈希」这件事
    跳过去了，而且空间固定浪费。用桶 + 链才是哈希表真正的样子，也解释了为什么
    平均 O(1)：只要键散得均匀，每条链都很短。

复杂度：平均时间 O(1)（链长为 1 + 装载因子），空间 O(n)。
"""


class MyHashMap:
    def __init__(self):
        self.size = 769
        self.buckets = [[] for _ in range(self.size)]

    def put(self, key, value):
        bucket = self.buckets[key % self.size]
        for pair in bucket:
            if pair[0] == key:
                pair[1] = value
                return
        bucket.append([key, value])

    def get(self, key):
        for k, v in self.buckets[key % self.size]:
            if k == key:
                return v
        return -1

    def remove(self, key):
        bucket = self.buckets[key % self.size]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return


if __name__ == "__main__":
    m = MyHashMap()
    m.put(1, 1)
    m.put(2, 2)
    assert m.get(1) == 1
    assert m.get(3) == -1
    m.put(2, 1)
    assert m.get(2) == 1
    m.remove(2)
    assert m.get(2) == -1
    m.remove(2)
    assert m.get(2) == -1
    m.put(769, 100)          # 与 key=0 同桶，检验链地址法
    m.put(0, 7)
    assert m.get(769) == 100
    assert m.get(0) == 7
    m.remove(769)
    assert m.get(769) == -1
    assert m.get(0) == 7
    print("design_hashmap: all tests passed")
