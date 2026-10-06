# 设计题：把基础结构拼成 O(1) 容器

设计题和前面各篇不太一样：它很少需要新算法，而是给你一个**容器**和一组操作，要求每个
操作都达到某个复杂度。真正的难点是——**单一数据结构往往只擅长一件事**：数组擅长随机访问、
哈希表擅长按键定位、链表擅长在已知节点处增删。于是设计题的核心手法就是「**组合**」：
把两三种基础结构拼在一起，让每个操作都用到最擅长它的那一种。

判断该拼哪些结构，可以逐条追问：「这个操作慢在哪？能不能换一个擅长它的结构来补？」
比如「按 key 查找」慢了就加哈希表，「随机取元素」慢了就加数组，「在中间增删」慢了就加链表。

本篇收 5 道经典题，按依赖关系由浅入深：先把三种**单一结构**各实现一遍——
**706** 用桶 + 链实现哈希表、**707** 用哨兵双向链表实现链表、**622** 用环形数组实现队列；
再用两道**组合题**体会拼装的威力——**146** 用「哈希表 + 双向链表」实现 LRU 缓存、
**380** 用「数组 + 哈希表（值→下标）」实现 O(1) 随机集合。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：桶 + 链实现哈希表 | 706. 设计哈希映射 | 简单 |
| 模式二：哨兵头尾的双向链表 | 707. 设计链表 | 中等 |
| 模式三：环形数组实现队列 | 622. 设计循环队列 | 中等 |
| 模式四：哈希表 + 双向链表 | 146. LRU 缓存 | 中等 |
| 模式五：数组 + 哈希表的下标映射 | 380. O(1) 时间插入、删除和获取随机元素 | 中等 |

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
