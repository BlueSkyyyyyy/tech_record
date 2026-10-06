"""460. LFU 缓存（LFU Cache）

题目：设计并实现最不经常使用（LFU）缓存，容量为 capacity。支持：
    get(key)：存在则返回其值（并让使用频次 +1），否则返回 -1；
    put(key, value)：存在则更新值（频次 +1）；不存在则插入，若已满，淘汰「使用频次
        最低」的键；频次相同时，淘汰「最久未使用」的那个。
要求 get 和 put 都是 O(1)。

思路（哈希表 + 「频次 → 有序桶」+ 维护最小频次）：
    LRU 只按「最近使用」排序，一条双向链表就够；LFU 多了一个「频次」维度：要先比频次，
    频次再相同才比最近使用。于是把节点按频次分组，每个频次对应一个「桶」，桶内部再按
    最近使用排序（新用的放一端，淘汰从另一端取）。

    需要三样东西：
    - `nodes`：key -> [value, freq]，负责按键 O(1) 找到值和当前频次；
    - `buckets`：freq -> 该频次的键（有序，按最近使用排列）；
    - `min_freq`：当前最小频次，淘汰时直接从 `buckets[min_freq]` 取最少用的键。

    「访问一个键」= 把它从旧频次的桶里摘下、放进 freq+1 的桶。若旧桶空了且旧频次正好
    等于 `min_freq`，说明最小频次整体上移一档，`min_freq += 1`。新插入的键频次为 1，
    最小频次重置为 1。

    Python 用 `OrderedDict` 当「有顺序的集合」：插入放尾部表示最近使用，淘汰取头部
    （`popitem(last=False)`）。

复杂度：get / put 时间 O(1)，空间 O(capacity)。
"""

from collections import OrderedDict


class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.min_freq = 0
        self.nodes = {}     # key -> [value, freq]
        self.buckets = {}   # freq -> OrderedDict(key -> None)，尾部最新、头部最旧

    def _touch(self, key):
        value, freq = self.nodes[key]
        del self.buckets[freq][key]
        if not self.buckets[freq]:
            del self.buckets[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        new_freq = freq + 1
        self.nodes[key][1] = new_freq
        self.buckets.setdefault(new_freq, OrderedDict())[key] = None

    def get(self, key):
        if key not in self.nodes:
            return -1
        value = self.nodes[key][0]
        self._touch(key)
        return value

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.nodes:
            self.nodes[key][0] = value
            self._touch(key)
            return
        if len(self.nodes) >= self.capacity:
            lru_key, _ = self.buckets[self.min_freq].popitem(last=False)
            del self.nodes[lru_key]
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
        self.nodes[key] = [value, 1]
        self.buckets.setdefault(1, OrderedDict())[key] = None
        self.min_freq = 1


if __name__ == "__main__":
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1        # key1 频次升到 2
    cache.put(3, 3)                 # 淘汰频次最低的 key2（频次 1）
    assert cache.get(2) == -1
    assert cache.get(3) == 3        # key3 频次升到 2
    cache.put(4, 4)                 # key1 与 key3 频次都是 2，淘汰更久没用的 key1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    zero = LFUCache(0)
    zero.put(1, 1)                  # 容量为 0，什么都不存也不报错
    assert zero.get(1) == -1
    print("lfu_cache: all tests passed")
