# 设计题：把基础结构拼成 O(1) 容器

设计题和前面各篇不太一样：它很少需要新算法，而是给你一个**容器**和一组操作，要求每个
操作都达到某个复杂度。真正的难点是——**单一数据结构往往只擅长一件事**：数组擅长随机访问、
哈希表擅长按键定位、链表擅长在已知节点处增删。于是设计题的核心手法就是「**组合**」：
把两三种基础结构拼在一起，让每个操作都用到最擅长它的那一种。

判断该拼哪些结构，可以逐条追问：「这个操作慢在哪？能不能换一个擅长它的结构来补？」
比如「按 key 查找」慢了就加哈希表，「随机取元素」慢了就加数组，「在中间增删」慢了就加链表。

本篇收 10 道经典题，按依赖关系由浅入深。前半程先把三种**单一结构**各实现一遍——
**706** 用桶 + 链实现哈希表、**707** 用哨兵双向链表实现链表、**622** 用环形数组实现队列；
再用两道**组合题**体会拼装的威力——**146** 用「哈希表 + 双向链表」实现 LRU 缓存、
**380** 用「数组 + 哈希表（值→下标）」实现 O(1) 随机集合。后半程继续沿着「组合」这条
线往外扩：**933** 用队列做时间窗口、**981** 用「哈希表 + 二分」查历史版本、**1146** 只记
变化点再二分求快照、**355** 用「哈希表 + 多路归并」拼出新闻流、**460** 在 LRU 上再加一个
频次维度做出 LFU。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：桶 + 链实现哈希表 | 706. 设计哈希映射 | 简单 |
| 模式二：哨兵头尾的双向链表 | 707. 设计链表 | 中等 |
| 模式三：环形数组实现队列 | 622. 设计循环队列 | 中等 |
| 模式四：哈希表 + 双向链表 | 146. LRU 缓存 | 中等 |
| 模式五：数组 + 哈希表的下标映射 | 380. O(1) 时间插入、删除和获取随机元素 | 中等 |
| 模式六：队列做滑动窗口 | 933. 最近的请求次数 | 简单 |
| 模式七：哈希表 + 按时间戳二分 | 981. 基于时间的键值存储 | 中等 |
| 模式八：只记变化点的快照 + 二分 | 1146. 快照数组 | 中等 |
| 模式九：哈希表 + 多路归并的时间线 | 355. 设计推特 | 中等 |
| 模式十：频次分桶的 LFU 缓存 | 460. LFU 缓存 | 困难 |

> 本篇的设计题会反复用到链表与哈希表，遇到读不顺的地方可回看 `06-linked-list` 篇的
> 「虚拟头结点」「双向链表」和 `02-hash` 篇的「按键定位」。155. 最小栈、208. 实现 Trie
> 也是设计题，分别见 `07-stack` 与 `17-trie` 篇。

---

## 模式一：桶 + 链实现哈希表

**适用信号**：题目让你「不用内建哈希表」实现一个键值映射，键的范围可控、操作只有
增删查。

**核心动作**：哈希表 = **散列函数** + **冲突处理**。用一个定长桶数组，位置取
`key % size`；同一个位置上的多个键用一条链（list / vector）串起来，查找时只扫这条链。

**为什么是平均 O(1)**：只要键散得均匀，每条链的平均长度就是「装载因子」级别的常数。
真正决定性能的是桶的数量：桶太少链就长、桶太多浪费空间。题目里键值范围到 10^6，取一个
质数（如 769）做桶数就足够。

### 706. 设计哈希映射（简单）

**题目**：不使用内建哈希表库，实现 `put(key, value)`、`get(key)`（不存在返回 -1）、
`remove(key)`。

**思路**：三个操作都先定位到桶，再在桶里线性找键。`put` 找到就改值、否则追加；
`get` 找到返回值；`remove` 找到就删。

```python
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
```

```cpp
class MyHashMap {
public:
    MyHashMap() : buckets_(kSize), size_(kSize) {}

    void put(int key, int value) {
        auto &bucket = buckets_[key % size_];
        for (auto &pair : bucket) {
            if (pair.first == key) {
                pair.second = value;
                return;
            }
        }
        bucket.emplace_back(key, value);
    }

    int get(int key) {
        for (const auto &pair : buckets_[key % size_]) {
            if (pair.first == key) {
                return pair.second;
            }
        }
        return -1;
    }

    void remove(int key) {
        auto &bucket = buckets_[key % size_];
        for (size_t i = 0; i < bucket.size(); ++i) {
            if (bucket[i].first == key) {
                bucket.erase(bucket.begin() + i);
                return;
            }
        }
    }

private:
    static const int kSize = 769;
    std::vector<std::vector<std::pair<int, int>>> buckets_;
    int size_;
};
```

- **复杂度**：平均时间 O(1)，空间 O(n)。
- **易错点**：桶数取质数（769）能减少聚集；`put` 更新已有键时不要重复插入；
  `remove` 不存在的键要静默返回，不能报错；注意 769 与 0 同桶，用来验证链地址法。
- **相似题**：380. O(1) 时间插入删除随机元素（换一种「哈希 + 数组」的组合，见本篇
  模式五）；208. 实现 Trie（用树结构而非散列做前缀映射，见 `17-trie` 篇）。

---

## 模式二：哨兵头尾的双向链表

**适用信号**：需要在**已知节点**处 O(1) 增删，且经常操作首尾。

**核心动作**：放两个不存真实数据的**哨兵**节点 `head`、`tail`，让它们互指，真实节点
永远夹在中间。这样头插、尾插、删首、删尾都变成同一套指针改写，边界消失。

**为什么要双向**：删除一个节点需要改它的前驱的 `next`，单向链表拿不到前驱，只能从头
再走一遍，就不是 O(1) 了。双向链表每个节点存 `prev` / `next`，删除时才真正 O(1)。

### 707. 设计链表（中等）

**题目**：实现 `get(index)`、`addAtHead`、`addAtTail`、`addAtIndex(index, val)`、
`deleteAtIndex(index)`。

**思路**：把所有「插入」都归约成 `addAtIndex`：`addAtHead` 就是在下标 0 插，
`addAtTail` 就是在下标 `size` 插。`_node_at` 把下标翻译成节点指针，并从靠近的一端
出发，最多走 `size/2` 步。

```python
class _Node:
    __slots__ = ("val", "prev", "next")

    def __init__(self, val=0):
        self.val = val
        self.prev = None
        self.next = None


class MyLinkedList:
    def __init__(self):
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _node_at(self, index):
        if index < 0 or index >= self.size:
            return None
        if index < self.size // 2:
            cur = self.head.next
            for _ in range(index):
                cur = cur.next
        else:
            cur = self.tail.prev
            for _ in range(self.size - 1 - index):
                cur = cur.prev
        return cur

    def get(self, index):
        node = self._node_at(index)
        return node.val if node else -1

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index < 0 or index > self.size:
            return
        nxt = self.tail if index == self.size else self._node_at(index)
        node = _Node(val)
        prev = nxt.prev
        node.prev = prev
        node.next = nxt
        prev.next = node
        nxt.prev = node
        self.size += 1

    def deleteAtIndex(self, index):
        node = self._node_at(index)
        if node is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
```

```cpp
struct Node {
    int val;
    Node *prev;
    Node *next;
    explicit Node(int v = 0) : val(v), prev(nullptr), next(nullptr) {}
};

class MyLinkedList {
public:
    MyLinkedList() {
        head_ = new Node();
        tail_ = new Node();
        head_->next = tail_;
        tail_->prev = head_;
        size_ = 0;
    }

    int get(int index) {
        Node *node = nodeAt(index);
        return node ? node->val : -1;
    }

    void addAtHead(int val) { addAtIndex(0, val); }

    void addAtTail(int val) { addAtIndex(size_, val); }

    void addAtIndex(int index, int val) {
        if (index < 0 || index > size_) {
            return;
        }
        Node *nxt = (index == size_) ? tail_ : nodeAt(index);
        Node *prev = nxt->prev;
        Node *node = new Node(val);
        node->prev = prev;
        node->next = nxt;
        prev->next = node;
        nxt->prev = node;
        ++size_;
    }

    void deleteAtIndex(int index) {
        Node *node = nodeAt(index);
        if (!node) {
            return;
        }
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
        --size_;
    }

private:
    Node *nodeAt(int index) {
        if (index < 0 || index >= size_) {
            return nullptr;
        }
        if (index < size_ / 2) {
            Node *cur = head_->next;
            for (int i = 0; i < index; ++i) {
                cur = cur->next;
            }
            return cur;
        }
        Node *cur = tail_->prev;
        for (int i = 0; i < size_ - 1 - index; ++i) {
            cur = cur->prev;
        }
        return cur;
    }

    Node *head_;
    Node *tail_;
    int size_;
};
```

- **复杂度**：`get` / `addAtIndex` / `deleteAtIndex` 时间 O(min(index, n-index))，
  空间 O(n)。
- **易错点**：下标边界要分清——`get` / `deleteAtIndex` 要求 `0 <= index < size`，
  而 `addAtIndex` 允许 `index == size`（尾插）；`addAtIndex` 里 `index == size` 时
  目标节点是 `tail` 哨兵而不是 `_node_at(size)`（那会返回 None）；改指针时先接好
  新节点的两条边，再断旧边。
- **相似题**：206. 反转链表、92. 反转链表 II（纯链表操作，见 `06-linked-list` 篇）；
  146. LRU 缓存（用同一套哨兵双向链表，见本篇模式四）。

---

## 模式三：环形数组实现队列

**适用信号**：容量固定、要求入队出队 O(1)、并且明确是「循环使用」的队列。

**核心动作**：定长数组 + 队首下标 `head` + 元素个数 `count`，下标用取模绕回。
新元素放 `(head + count) % k`，出队只挪 `head`。

**为什么用 `count` 而不用 `tail` 区分空满**：环形数组里，「空」和「满」都会让
`head == tail`（因为下标绕圈），单看两个下标分不出来。多存一个 `count`，空满一目了然：
`count == 0` 为空，`count == k` 为满。

### 622. 设计循环队列（中等）

**题目**：实现固定容量 `k` 的循环队列：`enQueue`、`deQueue`、`Front`、`Rear`、
`isEmpty`、`isFull`。

**思路**：四个读写口全部由 `head`、`count` 推出，不需要额外的 `tail`：

```python
class MyCircularQueue:
    def __init__(self, k):
        self.data = [0] * k
        self.capacity = k
        self.head = 0
        self.count = 0

    def enQueue(self, value):
        if self.count == self.capacity:
            return False
        self.data[(self.head + self.count) % self.capacity] = value
        self.count += 1
        return True

    def deQueue(self):
        if self.count == 0:
            return False
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return True

    def Front(self):
        return -1 if self.count == 0 else self.data[self.head]

    def Rear(self):
        if self.count == 0:
            return -1
        return self.data[(self.head + self.count - 1) % self.capacity]

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.capacity
```

```cpp
class MyCircularQueue {
public:
    explicit MyCircularQueue(int k)
        : data_(k, 0), capacity_(k), head_(0), count_(0) {}

    bool enQueue(int value) {
        if (count_ == capacity_) {
            return false;
        }
        data_[(head_ + count_) % capacity_] = value;
        ++count_;
        return true;
    }

    bool deQueue() {
        if (count_ == 0) {
            return false;
        }
        head_ = (head_ + 1) % capacity_;
        --count_;
        return true;
    }

    int Front() const {
        return count_ == 0 ? -1 : data_[head_];
    }

    int Rear() const {
        if (count_ == 0) {
            return -1;
        }
        return data_[(head_ + count_ - 1) % capacity_];
    }

    bool isEmpty() const { return count_ == 0; }

    bool isFull() const { return count_ == capacity_; }

private:
    std::vector<int> data_;
    int capacity_;
    int head_;
    int count_;
};
```

- **复杂度**：所有操作时间 O(1)，空间 O(k)。
- **易错点**：队尾下标是 `(head + count - 1) % k`，别漏掉 `- 1`；队空时 `Front` / `Rear`
  都返回 -1；每次取下标都要 `% capacity`，`Rear` 里的 `-1` 在 Python 中也要靠 `%`
  兜住（C++ 里先判空再算）。
- **相似题**：232. 用栈实现队列、225. 用队列实现栈（受限容器互相模拟，见 `07-stack` 篇）；
  933. 最近的请求次数（用普通队列做滑动窗口）。

---

## 模式四：哈希表 + 双向链表

**适用信号**：既需要「按键 O(1) 定位」，又需要「按顺序 O(1) 增删」，典型就是缓存
淘汰策略。

**核心动作**：哈希表负责定位节点，双向链表负责维护顺序。哈希表存 `key -> 节点`，
链表按「新→旧」串起来，头哨兵后面是最新、尾哨兵前面是最旧。查一次就把节点挪到头部，
超容就丢尾哨兵的前驱。

**为什么必须双向**：淘汰最旧、更新最近，都要求在已知节点处 O(1) 摘除；单向链表拿不到
前驱，摘不掉，所以链表一定要双向。另一点是哈希表里必须同时存 `key`——淘汰时是从链表
尾部拿到节点，要反查它的 key 才能从哈希表里删掉，节点上没存 key 就删不了。

### 146. LRU 缓存（中等）

**题目**：实现容量为 `capacity` 的 LRU 缓存，`get` / `put` 都要 O(1)：`get` 命中要
把键标为最近使用；`put` 插入新键时若超容，淘汰最久未使用的键。

**思路**：`_remove` 把节点从链上摘下，`_add_front` 把节点插到头部（最近使用端）。
`get` 命中后先摘再插到头部；`put` 命中已有键同理只更新值；插入新键若超容，先摘掉
`tail.prev`（最旧）并从哈希表删除，再在头部插入新节点。

```python
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
```

```cpp
struct Node {
    int key;
    int value;
    Node *prev;
    Node *next;
    Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
};

class LRUCache {
public:
    explicit LRUCache(int capacity) : capacity_(capacity) {
        head_ = new Node(0, 0);
        tail_ = new Node(0, 0);
        head_->next = tail_;
        tail_->prev = head_;
    }

    int get(int key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return -1;
        }
        Node *node = it->second;
        remove(node);
        addFront(node);
        return node->value;
    }

    void put(int key, int value) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            Node *node = it->second;
            node->value = value;
            remove(node);
            addFront(node);
            return;
        }
        if (static_cast<int>(cache_.size()) == capacity_) {
            Node *lru = tail_->prev;
            remove(lru);
            cache_.erase(lru->key);
            delete lru;
        }
        Node *node = new Node(key, value);
        cache_[key] = node;
        addFront(node);
    }

private:
    void remove(Node *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void addFront(Node *node) {
        node->next = head_->next;
        node->prev = head_;
        head_->next->prev = node;
        head_->next = node;
    }

    int capacity_;
    std::unordered_map<int, Node *> cache_;
    Node *head_;
    Node *tail_;
};
```

- **复杂度**：`get` / `put` 时间 O(1)，空间 O(capacity)。
- **易错点**：节点里要存 `key`，否则淘汰时无法从哈希表反删；`get` 只是一个「读」操作，
  但命中后也必须把它挪到头部（LRU 里「用过就算最近使用」）；`put` 已有键时要先改值
  再挪头，别走新建分支，否则会多出一个节点；`_remove` 和 `_add_front` 的指针顺序不能
  颠倒。
- **相似题**：707. 设计链表（本篇模式二，同一套哨兵双向链表）；460. LFU 缓存
  （再加上频次维度，是 LRU 的进阶版）；155. 最小栈（用辅助结构维护额外信息，见
  `07-stack` 篇）。

---

## 模式五：数组 + 哈希表的下标映射

**适用信号**：既要 O(1) 增删、又要在集合里**等概率随机**取一个元素。

**核心动作**：数组负责「随机取」（下标均匀）+ 哈希表记录「值 → 下标」负责「O(1) 定位」。
删除时不去搬移整段数组，而是把**末尾元素搬到待删位置**填坑，再 `pop` 末尾——因为集合
元素的顺序本来就不重要。

**为什么这样删除是 O(1)**：普通数组删中间元素要把它后面所有元素往前挪，是 O(n)。而
「用末尾元素填坑」只改两个位置：被删位置写入末尾值、末尾弹出，再更新哈希表里末尾值
的新下标。整个过程中没有循环，所以 O(1)。

### 380. O(1) 时间插入、删除和获取随机元素（中等）

**题目**：实现集合的 `insert(val)`、`remove(val)`、`getRandom()`，都在平均 O(1)。

**思路**：`insert` 先查重再追加并记下标；`remove` 用末尾元素填坑后弹出；`getRandom`
随机取一个下标直接返回。

```python
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
```

```cpp
class RandomizedSet {
public:
    RandomizedSet() { std::srand(12345); }

    bool insert(int val) {
        if (pos_.count(val)) {
            return false;
        }
        pos_[val] = static_cast<int>(vals_.size());
        vals_.push_back(val);
        return true;
    }

    bool remove(int val) {
        auto it = pos_.find(val);
        if (it == pos_.end()) {
            return false;
        }
        int idx = it->second;
        int last = vals_.back();
        vals_[idx] = last;
        pos_[last] = idx;
        vals_.pop_back();
        pos_.erase(val);
        return true;
    }

    int getRandom() {
        return vals_[std::rand() % vals_.size()];
    }

private:
    std::vector<int> vals_;
    std::unordered_map<int, int> pos_;
};
```

- **复杂度**：三个操作平均时间 O(1)，空间 O(n)。
- **易错点**：`remove` 里若被删的正好是末尾元素，`last` 就是 `val` 本身，先把
  `pos[last] = idx` 再 `erase(val)` 也没问题，但顺序不能反（先删了 `val` 再设
  `pos[last]` 会把刚删的键又加回来）；`insert` 必须先判重，否则数组和下标表会不一致。
- **相似题**：381. O(1) 时间插入删除随机元素 II（允许重复元素，哈希表里存下标集合，
  见同名题的进阶版）；706. 设计哈希映射（本篇模式一，同样用「值→位置」的反查思想）。

---

## 模式六：队列做滑动窗口

**适用信号**：数据按时间（或顺序）**单调到来**，每次只问「最近一段时间内」的统计量。

**核心动作**：把到来的元素依次放进队列（FIFO）。每次先把队首所有「已经过期」的元素
弹出，再读队列长度。窗口右端一直跟着新元素走，左端只往后移，所以每个元素最多进出队
一次。

**为什么用队列**：这里的窗口是时间维度上宽度固定的一段，元素一个接一个到来，天然是
「先进先出」。它和 `03-sliding-window` 篇是同一个思想，只是那里的窗口按元素个数或
条件伸缩，这里按时间边界切。

### 933. 最近的请求次数（简单）

**题目**：实现 `ping(t)`：在时间 `t` 新增一次请求（`t` 单调递增），返回闭区间
`[t - 3000, t]` 内的请求次数。

**思路**：请求时间单调递增，所以过期的一定在队首。每次 `ping` 先入队，再把队首所有
小于 `t - 3000` 的时间弹出，剩下的个数就是窗口内的请求数。

```python
from collections import deque


class RecentCounter:
    def __init__(self):
        self.q = deque()

    def ping(self, t):
        self.q.append(t)
        while self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)
```

```cpp
class RecentCounter {
public:
    int ping(int t) {
        q_.push_back(t);
        while (q_.front() < t - 3000) {
            q_.pop_front();
        }
        return static_cast<int>(q_.size());
    }

private:
    std::deque<int> q_;
};
```

- **复杂度**：每次 `ping` 摊还时间 O(1)，空间 O(窗口内请求数)。
- **易错点**：窗口是**闭区间**，边界 `t - 3000` 本身要算在内，所以弹出条件是
  `< t - 3000`（不能写 `<=`）；用队列而不是普通数组，弹出队首才方便。
- **相似题**：622. 设计循环队列（本篇模式三，同样是 FIFO 结构）；3. 无重复字符的
  最长子串、209. 长度最小的子数组（元素维度的滑动窗口，见 `03-sliding-window` 篇）。

---

## 模式七：哈希表 + 按时间戳二分

**适用信号**：同一个键会被反复写入，每次带一个**递增的时间戳**，查询时要「取某个时间
点上的值」。

**核心动作**：键用哈希表定位，值不再只存一份，而是存一串**按时序排列的历史**。查询时
对历史里的时间戳做二分，找到最后一个不超过给定时间的那一版。

**为什么能这么存**：题目保证同一键的写入时间戳严格递增，所以历史天然有序，直接追加
即可，查询用 `bisect_right` 一步定位。

### 981. 基于时间的键值存储（中等）

**题目**：实现 `set(key, value, timestamp)` 与 `get(key, timestamp)`。`get` 返回该键
在「不晚于 `timestamp` 的最近一次 `set`」里的值，不存在则返回空串。

**思路**：`times` 和 `values` 两张表按 key 各自维护平行的历史列表（一张存时间戳、一张
存值）。`get` 在 `times` 上做一次 `bisect_right` 得到第一个大于 `timestamp` 的下标，
减一就是答案位置；为负说明查询时间早于首次 `set`，返回空串。

```python
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
```

```cpp
class TimeMap {
public:
    void set(const std::string &key, const std::string &value, int timestamp) {
        times_[key].push_back(timestamp);
        values_[key].push_back(value);
    }

    std::string get(const std::string &key, int timestamp) {
        auto it = times_.find(key);
        if (it == times_.end()) {
            return "";
        }
        const std::vector<int> &ts = it->second;
        int i = static_cast<int>(
                    std::upper_bound(ts.begin(), ts.end(), timestamp) - ts.begin()) -
                1;
        return i >= 0 ? values_[key][i] : "";
    }

private:
    std::unordered_map<std::string, std::vector<int>> times_;
    std::unordered_map<std::string, std::vector<std::string>> values_;
};
```

- **复杂度**：`set` 时间 O(1)；`get` 时间 O(log n)（n 为该键的历史次数），
  空间 O(总 `set` 次数)。
- **易错点**：要找的是「最后一个 `<= timestamp`」，所以用 `bisect_right`（第一个
  `> timestamp` 的下标）再减一，而不是 `bisect_left`；下标为负时要返回空串，不能直接
  索引。两张表用同一种顺序追加，保证下标一一对应。
- **相似题**：1146. 快照数组（本篇模式八，同一套「版本历史 + 二分」）；35. 搜索插入
  位置、34. 查找区间（`bisect_right` / `upper_bound` 的模板，见 `05-binary-search` 篇）。

---

## 模式八：只记变化点的快照 + 二分

**适用信号**：要求支持「拍快照」并「读历史快照里的值」，且快照次数很多、数组很大。

**核心动作**：不要每次快照都复制整个数组（那是「快照数 × 长度」的空间）。改为**每个
下标单独记一行 `(快照号, 值)` 的历史**，只在值真的变化时追加一条。查询某个下标在某个
快照号的值时，就在这一行里二分找最后一条快照号不超过它的记录。

**为什么省空间**：相邻两次快照之间，通常只有极少数下标被改过，重复的下标不必反复记录。
空间正比于「发生过的修改次数」，而不是「快照数 × 数组长度」。

### 1146. 快照数组（中等）

**题目**：实现 `set(index, val)`、`snap()`（返回快照号并递增）、`get(index, snap_id)`。
未 `set` 过的位置默认值为 0。

**思路**：每行历史开头放一个哨兵 `(-1, 0)`，让任意合法快照号都能查到结果。`set` 时若
当前快照号已经有记录就覆盖那一条，否则追加（避免同一快照号堆多条）。`get` 用手写二分
找最后一条快照号 `<= snap_id` 的记录。

```python
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
```

```cpp
class SnapshotArray {
public:
    explicit SnapshotArray(int length)
        : history_(length, std::vector<std::pair<int, int>>{{-1, 0}}), snap_id_(0) {}

    void set(int index, int val) {
        std::vector<std::pair<int, int>> &row = history_[index];
        if (row.back().first == snap_id_) {
            row.back().second = val;
        } else {
            row.emplace_back(snap_id_, val);
        }
    }

    int snap() { return snap_id_++; }

    int get(int index, int snap_id) {
        const std::vector<std::pair<int, int>> &row = history_[index];
        int lo = 0, hi = static_cast<int>(row.size()) - 1, ans = 0;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (row[mid].first <= snap_id) {
                ans = row[mid].second;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    }

private:
    std::vector<std::vector<std::pair<int, int>>> history_;
    int snap_id_;
};
```

- **复杂度**：`set` 均摊 O(1)，`snap` O(1)，`get` O(log m)（m 为该下标的历史修改
  次数）；空间 O(总修改次数 + length)。
- **易错点**：`snap()` 是「先返回当前快照号、再自增」，所以快照号从 0 开始；`set` 必须
  判断「当前快照号是否已有记录」，否则在同一快照号里 `set` 两次会留下两条记录，二分时
  可能取到旧值；哨兵 `(-1, 0)` 保证没改过的下标也能返回默认值 0。
- **相似题**：981. 基于时间的键值存储（本篇模式七，都是「历史 + 二分」）；704. 二分
  查找（手写二分的闭区间模板，见 `05-binary-search` 篇）。

---

## 模式九：哈希表 + 多路归并的时间线

**适用信号**：要在多个「按时间排列的列表」里合并出最近的前 k 条。

**核心动作**：把每个来源的列表看成一链有序数据，链尾是最新的。用一个堆，初始把每条链
的最新一条压进去；每次弹出时间最大的那条加入结果，再从它所属的链里补上前一条，直到凑
满 k 条。这样只碰每链末尾的少量元素，不必把所有数据全排序。

**为什么用堆**：k 条链的「当前最新」是 k 个候选，每次选出最大的那个就是一次多路归并。
堆的大小只有「来源数」，比把所有推文合在一起排序省得多。

### 355. 设计推特（中等）

**题目**：实现 `postTweet`、`getNewsFeed`（返回自己与关注者最近 10 条推文 id，最近的
在前）、`follow`、`unfollow`。

**思路**：全局递增时间戳保证跨用户可比。每个用户维护一个按时间递增的推文列表；关注关系
用集合（保证不重复、取关能删）。`getNewsFeed` 取「自己 + 所有关注者」的列表做多路
归并，用小顶堆配合取负的时间戳选出最新的那条。

```python
import heapq


class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = {}       # user -> [(time, tweetId), ...]，时间递增
        self.following = {}    # user -> set(followee)

    def postTweet(self, userId, tweetId):
        self.time += 1
        self.tweets.setdefault(userId, []).append((self.time, tweetId))

    def getNewsFeed(self, userId):
        sources = [self.tweets.get(userId, [])]
        for followee in self.following.get(userId, ()):
            if followee != userId:
                sources.append(self.tweets.get(followee, []))

        heap = []
        for i, lst in enumerate(sources):
            if lst:
                heap.append((-lst[-1][0], i, len(lst) - 1))
        heapq.heapify(heap)

        feed = []
        while heap and len(feed) < 10:
            _, i, j = heapq.heappop(heap)
            feed.append(sources[i][j][1])
            if j > 0:
                heapq.heappush(heap, (-sources[i][j - 1][0], i, j - 1))
        return feed

    def follow(self, followerId, followeeId):
        self.following.setdefault(followerId, set()).add(followeeId)

    def unfollow(self, followerId, followeeId):
        if followerId in self.following:
            self.following[followerId].discard(followeeId)
```

```cpp
class Twitter {
public:
    void postTweet(int userId, int tweetId) {
        ++time_;
        tweets_[userId].emplace_back(time_, tweetId);
    }

    std::vector<int> getNewsFeed(int userId) {
        std::vector<const std::vector<std::pair<int, int>> *> sources;
        auto addSource = [&](int uid) {
            auto it = tweets_.find(uid);
            if (it != tweets_.end() && !it->second.empty()) {
                sources.push_back(&it->second);
            } else {
                sources.push_back(nullptr);
            }
        };
        addSource(userId);
        auto fit = following_.find(userId);
        if (fit != following_.end()) {
            for (int followee : fit->second) {
                if (followee != userId) {
                    addSource(followee);
                }
            }
        }

        std::priority_queue<std::tuple<int, int, int>> pq;
        for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
            if (sources[i]) {
                int j = static_cast<int>(sources[i]->size()) - 1;
                pq.emplace((*sources[i])[j].first, i, j);
            }
        }

        std::vector<int> feed;
        while (!pq.empty() && static_cast<int>(feed.size()) < 10) {
            auto [ts, i, j] = pq.top();
            pq.pop();
            feed.push_back((*sources[i])[j].second);
            if (j > 0) {
                pq.emplace((*sources[i])[j - 1].first, i, j - 1);
            }
        }
        return feed;
    }

    void follow(int followerId, int followeeId) {
        following_[followerId].insert(followeeId);
    }

    void unfollow(int followerId, int followeeId) {
        auto it = following_.find(followerId);
        if (it != following_.end()) {
            it->second.erase(followeeId);
        }
    }

private:
    int time_ = 0;
    std::unordered_map<int, std::vector<std::pair<int, int>>> tweets_;  // user -> (time, id)
    std::unordered_map<int, std::unordered_set<int>> following_;        // user -> followees
};
```

- **复杂度**：`postTweet` / `follow` / `unfollow` 时间 O(1)；`getNewsFeed` 时间
  O((F + 10) log F)（F 为关注数），空间 O(总推文数)。
- **易错点**：Python 的 `heapq` 是小顶堆，想取最新（时间最大）就得把时间取负；C++ 的
  `priority_queue` 默认是大顶堆，直接放正的时间即可——两种语言的取号方向相反，别抄混。
  堆里要同时带「哪条链、链内下标」才能弹出后回补；自己关注自己时，构建来源列表要去重，
  否则自己的推文会出现两遍。取关不存在的关注是静默无操作。
- **相似题**：23. 合并 K 个升序链表、373. 查找和最小的 K 对（多路归并的模板，见
  `08-heap` 篇）；146. LRU 缓存（本篇模式四，哈希表 + 链表的组合思路）。

---

## 模式十：频次分桶的 LFU 缓存

**适用信号**：淘汰策略是「用得最少的先淘汰」，用得一样少时再比「谁更久没用」。

**核心动作**：在 LRU 的基础上多一层**频次**维度。把键按频次分到不同的桶里，每个桶内部
仍像 LRU 一样按「最近使用」排序；再单独维护一个 `min_freq`，淘汰时直接从
`buckets[min_freq]` 里取最少用的键。访问一个键时，把它从旧频次的桶搬到 `freq + 1` 的
桶，并在旧桶空了且正好是最小频次时把 `min_freq` 加一。

**为什么能 O(1)**：按键定位靠哈希表；桶内是双向链表 / 有序字典，摘除和插入都 O(1)；
`min_freq` 只会「在最小桶被清空时上移一档」，访问一次最多加一，所以不需要扫描就能知道
当前谁最少用。

### 460. LFU 缓存（困难）

**题目**：设计 LFU 缓存，容量为 `capacity`。`get` / `put` 都要 O(1)：`get` 命中让频次
加一；`put` 插入新键时若超容，淘汰频次最低的键，频次相同则淘汰最久未使用的。

**思路**：三样东西配合——`nodes` 记 `key -> [value, freq]`；`buckets` 记
`freq -> 有序键集`（尾部最新、头部最旧）；`min_freq` 记当前最小频次。`_touch` 负责
「摘旧桶、放新桶、按需上移 `min_freq`」。Python 用 `OrderedDict` 当有序集合，
`popitem(last=False)` 取最久未用的键。

```python
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
```

```cpp
class LFUCache {
public:
    explicit LFUCache(int capacity) : capacity_(capacity), min_freq_(0) {}

    int get(int key) {
        auto it = info_.find(key);
        if (it == info_.end()) {
            return -1;
        }
        int value = it->second.first;
        touch(key);
        return value;
    }

    void put(int key, int value) {
        if (capacity_ <= 0) {
            return;
        }
        auto it = info_.find(key);
        if (it != info_.end()) {
            it->second.first = value;
            touch(key);
            return;
        }
        if (static_cast<int>(info_.size()) >= capacity_) {
            std::list<int> &bucket = buckets_[min_freq_];
            int evict = bucket.back();          // 该频次里最久未使用
            bucket.pop_back();
            pos_.erase(evict);
            info_.erase(evict);
            if (bucket.empty()) {
                buckets_.erase(min_freq_);
            }
        }
        info_[key] = {value, 1};
        buckets_[1].push_front(key);
        pos_[key] = buckets_[1].begin();
        min_freq_ = 1;
    }

private:
    void touch(int key) {
        int freq = info_[key].second;
        auto &bucket = buckets_[freq];
        bucket.erase(pos_[key]);
        pos_.erase(key);
        if (bucket.empty()) {
            buckets_.erase(freq);
            if (min_freq_ == freq) {
                ++min_freq_;
            }
        }
        int new_freq = freq + 1;
        info_[key].second = new_freq;
        buckets_[new_freq].push_front(key);
        pos_[key] = buckets_[new_freq].begin();
    }

    int capacity_;
    int min_freq_;
    std::unordered_map<int, std::pair<int, int>> info_;       // key -> (value, freq)
    std::unordered_map<int, std::list<int>> buckets_;         // freq -> keys
    std::unordered_map<int, std::list<int>::iterator> pos_;   // key -> 在桶里的位置
};
```

- **复杂度**：`get` / `put` 时间 O(1)，空间 O(capacity)。
- **易错点**：`min_freq` 只在「被搬空的旧桶正好是最小频次桶」时才加一，别无条件加；
  新键插入后 `min_freq` 一定要重置为 1（哪怕缓存里还有高频键）；`capacity == 0` 时
  `put` 直接返回、`get` 返回 -1；C++ 用 `list` 的迭代器定位节点，摘除后别忘同步
  `pos_`。频次相同时按「最久未使用」淘汰，所以桶内要维护最近使用顺序（插入放前端、
  淘汰取后端）。
- **相似题**：146. LRU 缓存（本篇模式四，LFU 的简化版，只差频次维度）；
  380. O(1) 时间插入、删除和获取随机元素（本篇模式五，同样是「哈希表 + 另一种结构」
  的组合）。

---

## 规律总结

1. **设计题 = 选结构 + 组合**。拿到题先逐条列出每个操作的复杂度要求，再问「哪个结构
   天生擅长这件事」：按键定位找哈希表、随机访问找数组、已知节点增删找链表、
   维护顺序找双向链表。
2. **单一结构的短板，用另一个结构来补**。LRU 是「哈希表补链表的定位、链表补哈希表的
   顺序」；随机集合是「数组补随机、哈希表补定位」。组合时两边的信息要同步更新
   （例如填坑后必须改哈希表里的下标）。
3. **哨兵节点消除边界**。头尾各放一个不存数据的哨兵，头插 / 尾插 / 删除全部统一成
   同一套指针改写。需要 O(1) 摘除时，链表一定要**双向**，因为删除依赖前驱。
4. **数组删除的 O(1) 技巧是「末尾填坑」**。只要元素顺序无关紧要，删中间元素时用末尾
   元素覆盖它再 `pop`，就能把 O(n) 降到 O(1)，代价是维护一张「值→下标」的表。
5. **环形数组要单独记一个 `count`**。环形里空和满都会让首尾下标相等，只有元素个数能
   区分它们；下标一律用取模绕回。
6. **操作即语义**：`get` 在 LRU 里不只是读，还是「标记最近使用」；`remove` 不存在时
   要静默返回。设计题照着题面给的语义写，不要按直觉多加或少做动作。
7. **带「时间」的操作先想清时间的用法**：固定长度的时间窗口用队列维护（933）；要按
   时间点查历史版本，就把版本号排成有序序列再用二分（981 / 1146）。前者是「只留窗口
   内的」，后者是「历史全留、按版本取」。
8. **版本 / 快照题的核心是「只记变化点」**。别每次快照都抄一份全量数据，把每次修改按
   版本号追加成历史，查询时二分定位，空间就从「快照数 × 长度」降到「修改次数」。
   注意同一版本内的重复修改要覆盖而不是追加。
9. **LFU = LRU + 一个频次维度**。把键按频次分桶、桶内仍按最近使用排序，再单独维护
   `min_freq`，就能 O(1) 选出淘汰对象。多一个排序维度，就多一层分桶。
10. **「取最近 k 条」用多路归并**。每个来源的有序列表当作一条链，堆里只放各链当前的
    最新元素，弹出一条再补它前一条，堆的大小只与来源数有关，不必把所有数据合起来排序。
