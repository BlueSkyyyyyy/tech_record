"""146. LRU 缓存（LRU Cache）

题目：设计一个满足 LRU（最近最少使用）策略的缓存，容量为 capacity。支持：
    get(key)：键存在则返回其值并把该键标记为「最近使用」，否则返回 -1；
    put(key, value)：键存在则更新并标记为最近使用；不存在则插入，若已超出容量，
        淘汰最久未使用的键。

要求 get 和 put 都是 O(1)。

思路（哈希表 + 双向链表）：
    两个操作各需要一种能力，单靠一种结构都不够：
    - 按 key 定位节点的值和位置 → 需要哈希表 O(1)；
    - 在 O(1) 里把某个节点挪到「最新端」、并从「最旧端」丢掉一个 → 需要能 O(1)
      摘除和插入的链表，且要双向（单向拿不到前驱，删不掉）。

    于是把二者拼起来：哈希表 `key -> 节点`，链表把节点按「使用新→旧」串起来。
    再加头尾两个哨兵，头哨兵后面就是最新、尾哨兵前面就是最旧。
    - get：哈希查到节点，把它从链上摘下再插到头部；
    - put：键已存在就改值并挪到头部；否则新建节点插头部，并写进哈希；若超容，
      就删掉尾哨兵的前驱（最旧节点），同时从哈希里删掉它的 key。

    摘除 + 插入要用「先连新边、再断旧边」的顺序写，避免指针丢失。

复杂度：get / put 时间 O(1)，空间 O(capacity)。
"""


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key, value):
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return
        if len(self.cache) == self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
        node = _Node(key, value)
        self.cache[key] = node
        self._add_front(node)


if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1        # 1 变为最近使用，2 变最旧
    cache.put(3, 3)                 # 淘汰 2
    assert cache.get(2) == -1
    assert cache.get(3) == 3
    cache.put(4, 4)                 # 淘汰 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    cache.put(3, 30)                # 更新 3 的值
    assert cache.get(3) == 30
    cache.put(5, 5)                 # 淘汰最旧的 4
    assert cache.get(4) == -1
    assert cache.get(3) == 30
    assert cache.get(5) == 5
    small = LRUCache(1)
    small.put(1, 1)
    small.put(2, 2)
    assert small.get(1) == -1
    assert small.get(2) == 2
    print("lru_cache: all tests passed")
