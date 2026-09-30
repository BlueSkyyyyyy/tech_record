# 堆与 Top-K

堆（heap，又叫优先队列 priority queue）是一种能**快速取出最值**的数据结构：
插入、删除、取最值都是 O(log n)。它最常见的用途是解决一类叫 **Top-K** 的问题——
「最大的 k 个」「频率最高的 k 个」「第 k 大」。这类问题的共同点是：
我们并不需要把全部数据排好序，只关心**最靠前的一小撮**。

堆还有第二类高频用途：**多路归并**——把多条各自有序的序列合并成一条，
每次都从「各序列的当前头」里取最小（或最大）。合并 K 个有序链表、有序矩阵第 K 小、
两个数组的最小 K 对，本质都是这一招。

此外，堆还能处理**动态最值**：数据源源不断到来（数据流），却要随时回答
「当前第 K 大」「当前中位数」。这时堆是少数能同时做到 O(log n) 更新、O(1) 查询的结构。

> 说明：Python 的 `heapq` 只提供**小顶堆**；C++ 的 `priority_queue` 默认是大顶堆，
> 想用小顶堆要额外传比较器 `std::greater<T>`。这个差别是本题型最容易踩的坑。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 小顶堆当淘汰线 | 215. 数组中的第 K 个最大元素 | 中等 |
| 小顶堆当淘汰线 | 347. 前 K 个高频元素 | 中等 |
| 数据流里滚动维护 | 703. 数据流中的第 K 大元素 | 简单 |
| 数据流里滚动维护 | 295. 数据流的中位数 | 困难 |
| 多路归并 | 23. 合并 K 个升序链表 | 困难 |
| 多路归并 | 378. 有序矩阵中第 K 小的元素 | 中等 |
| 多路归并 | 373. 查找和最小的 K 对数字 | 中等 |
| 大顶堆模拟 | 1046. 最后一块石头的重量 | 简单 |
| 自定义优先级 | 973. 最接近原点的 K 个点 | 中等 |
| 自定义优先级 | 692. 前 K 个高频单词 | 中等 |

---

## 模式一：用小顶堆当淘汰线

**适用信号**：题目要求「最大/最小/最高频的 k 个」，且 k 通常远小于 n。
关键词是「第 K 个」「前 K 个」「Top K」。核心动作：维护一个**大小为 k 的小顶堆**，
堆顶就是当前集合里最「不够格」的那个；新元素只和堆顶比一次，就能决定留不留。

为什么是小顶堆而不是大顶堆：求 Top K 大，我们希望**随时能踢掉最小的那个**，
好让位置留给更大的新元素。小顶堆的堆顶恰好就是最小，淘汰它只需 O(log k)，无需遍历。

### 215. 数组中的第 K 个最大元素（中等）

**题目**：给定整数数组 `nums` 和整数 `k`，返回数组中第 `k` 个最大的元素。注意是排序后的第 `k` 个最大，而不是第 `k` 个不同的元素。

**思路（维护一个大小为 k 的小顶堆）**：
遍历数组，把每个数压入一个小顶堆（堆顶最小）。一旦堆的大小超过 `k`，就弹出堆顶——被弹掉的永远是「当前已见过的数里最小的那个」。遍历结束时，堆里留下的是整个数组中最大的 `k` 个数，而堆顶就是这 `k` 个里最小的，也就是全局第 `k` 大。

为什么用「小顶堆 + 限制大小 k」：我们要的是第 `k` 大，相当于只要保留最大的 `k` 个就够了，多余的小数直接扔掉。小顶堆的堆顶天生就是「当前保留集合里最小的」，正好是淘汰线：新来一个数只要比堆顶大，就值得挤进来，把堆顶踢掉；比堆顶小则连进都不用进。这样堆里始终只有 `k` 个元素，时间和空间都只与 `k` 有关，而不是与 `n` 有关。

为什么不用大顶堆一次全装进去再弹 k 次：那样要先把 n 个元素都建堆，再弹 k 次，复杂度 O(n + k log n)，通常比 O(n log k) 慢，而且浪费了「只需要 k 个」这一信息。另一种做法是快速选择（quickselect），平均 O(n)，但最坏 O(n²)、实现更容易写错，这里只详展最稳妥的堆解。

**代码**（完整可运行版见 `src/heap/kth_largest_element_in_an_array.py` / `.cpp`）：

```python
import heapq


def find_kth_largest(nums, k):
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]
```

```cpp
int findKthLargest(const std::vector<int> &nums, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}
```

- **复杂度**：时间 O(n log k)（每个元素入堆一次，堆高为 log k），空间 O(k)。
- **易错点**：Python 用 `heapq`（小顶堆）即可；C++ 的 `priority_queue` 默认是**大顶堆**，必须写成 `priority_queue<int, vector<int>, greater<int>>` 才是小顶堆，写错会变成保留最大的 k 个却返回堆顶最大值，答案相反；越界判断是 `size() > k`（超过才弹），不是 `>=`；返回的是 `min_heap[0]` / `minHeap.top()`，不是弹出的值。
- **相似题**：347. 前 K 个高频元素（同一模板，排序依据换成频率，见下）；973. 最接近原点的 K 个点（求最小，改用大顶堆，见「模式五」）；703. 数据流中的第 K 大元素（把一次性的数组换成持续到来的数据流，见「模式二」）；215 的快速选择解常在分治篇里再讲。

### 347. 前 K 个高频元素（中等）

**题目**：给定一个整数数组 `nums` 和一个整数 `k`，返回其中出现频率前 `k` 高的元素。可以按任意顺序返回答案。

**思路（先统频率，再用大小为 k 的小顶堆筛出前 k）**：
先用哈希表统计每个元素的出现次数，得到「元素 → 频率」。
然后遍历这张表，把 `(频率, 元素)` 压入一个小顶堆，堆顶是频率最小的那对。
一旦堆的大小超过 `k`，就弹出堆顶，把频率最低的淘汰掉。
遍历结束时，堆里剩下的就是频率最高的 `k` 个元素。

为什么是「频率入堆」而不是「元素入堆」：排序的依据是频率，但输出的是元素，所以把两者打包成一个元组一起进堆，让堆按频率自动排序。Python 的元组比较先比频率、频率相同时再比元素，这只是一个稳定的兜底规则，不影响正确性。C++ 用 `pair<int,int>`，比较规则同理（先比 `first`）。

为什么又是「小顶堆 + 限制大小 k」：和 215 完全同构——要 Top K 大的，就保留一个大小为 k 的小顶堆，让堆顶当淘汰线，每次只留最有资格的 k 个。堆里元素数始终是 k，和「总共有多少种不同元素」无关，比把所有元素排序再取前 k 快得多。

另一种做法是桶排序：频率最大不超过 n，用「频率 → 元素列表」的数组，从高到低扫桶、取满 k 个即可，时间 O(n)，但需要 O(n) 额外空间。这里详展通用、好写的堆解。

**代码**（`src/heap/top_k_frequent_elements.py` / `.cpp`）：

```python
import heapq


def top_k_frequent(nums, k):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    min_heap = []
    for num, freq in count.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for freq, num in min_heap]
```

```cpp
std::vector<int> topKFrequent(const std::vector<int> &nums, int k) {
    std::unordered_map<int, int> count;
    for (int num : nums) ++count[num];

    using P = std::pair<int, int>;  // (频率, 元素)
    std::priority_queue<P, std::vector<P>, std::greater<P>> minHeap;
    for (const auto &[num, freq] : count) {
        minHeap.push({freq, num});
        if ((int)minHeap.size() > k) minHeap.pop();
    }

    std::vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }
    return result;
}
```

- **复杂度**：时间 O(n + m log k)（n 为数组长度，m 为不同元素个数），空间 O(m + k)。
- **易错点**：元组/pair 的顺序必须是 `(频率, 元素)`，写反会变成按元素值筛、结果全错；C++ 小顶堆同样要显式传 `greater<pair<int,int>>`；返回元素时 `pair` 取 `second`（第二项），别取 `first`（那是频率）；答案顺序任意，比较时先排序再比，否则测试会因顺序不同而误判。
- **相似题**：692. 前 K 个高频单词（同题加一个「同频按字典序」的比较规则，见「模式五」）；215. 数组中的第 K 个最大元素（同一「小顶堆 + 限制 k」模板，见上）；451. 根据字符出现频率排序（统计频率后的另一种应用）。

---

## 模式二：数据流里滚动维护

**适用信号**：数据不是一次性给全，而是**持续到来**，却要随时能回答「当前的第 K 大」
或「当前的中位数」。这时没法先排序，只能用**固定规模的堆**滚动维护答案。
703 是 215 的在线版；295 则用两个堆分别托住「较小的一半」和「较大的一半」。

### 703. 数据流中的第 K 大元素（简单）

**题目**：设计一个类 `KthLargest`，初始化时给定 `k` 和一个初始数组 `nums`；之后每次调用 `add(val)` 把 `val` 加入数据流，并返回当前数据流中第 `k` 大的元素。

**思路（维护一个大小为 k 的小顶堆，边来边更新）**：
和 215 是同一个套路，区别在于数据不是一次性给全，而是「持续到来」。构造时先把 `nums` 里的每个数都 `add` 一遍。每次 `add(val)`：把 `val` 压入小顶堆，再限制堆大小不超过 `k`，超过就弹堆顶。堆顶就是当前保留的 k 个数里最小的那个，也就是当前数据流的第 k 大。

为什么这样能在线维护：我们永远只需要「最大的 k 个数」，把它们放在一个小顶堆里，堆顶就是淘汰线。新数进来只和堆顶比一次：比堆顶大就挤掉堆顶、自己留下；比堆顶小就不用留。这样无论数据来多少，堆始终只有 `k` 个元素，内存不随数据总量增长。

为什么要保留重复元素：题目按「排序后的第 k 个」计数，不去重。所以同一个值出现多次，就要在堆里占多个位置，不能当集合处理。

**代码**（`src/heap/kth_largest_element_in_a_stream.py` / `.cpp`）：

```python
import heapq


class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)

    def add(self, val):
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

```cpp
class KthLargest {
public:
    KthLargest(int k, const std::vector<int> &nums) : k_(k) {
        for (int num : nums) add(num);
    }

    int add(int val) {
        minHeap_.push(val);
        if ((int)minHeap_.size() > k_) minHeap_.pop();
        return minHeap_.top();
    }

private:
    int k_;
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap_;
};
```

- **复杂度**：构造 O(n log k)；单次 `add` 时间 O(log k)；空间 O(k)。
- **易错点**：初始化时不能只把 `nums` 直接丢进堆，必须走 `add`（或等价的「压入后限制到 k」），否则堆里可能留下超过 k 个、数量最小的元素；`add` 要**返回堆顶**而不是弹出的值；C++ 的构造函数里调 `add` 前，成员 `k_` 必须已经初始化（初始化列表保证顺序）。
- **相似题**：215. 数组中的第 K 个最大元素（离线版，见「模式一」）；295. 数据流的中位数（在线版的进阶，见下）；347 的前 K 高频若换成流式统计，也是同样的滚动思路。

### 295. 数据流的中位数（困难）

**题目**：设计一个数据结构，支持两种操作：`addNum(num)` 从数据流中添加一个整数；`findMedian()` 返回当前所有元素的中位数。偶数个元素时中位数取中间两个数的平均值。

**思路（对顶堆：左半边用大顶堆，右半边用小顶堆）**：
把已经读到的数分成两堆：`left` 是较小的一半，用大顶堆，堆顶是这半边最大的数；`right` 是较大的一半，用小顶堆，堆顶是这半边最小的数。只要保证两个条件，中位数就唾手可得：

1. `left` 的元素个数与 `right` 相等或恰好比 `right` 多 1；
2. `left` 里每个数都不大于 `right` 里每个数。

此时：元素总数为奇数，中位数就是 `left` 堆顶；总数为偶数，中位数就是 `(left 堆顶 + right 堆顶) / 2`。

每次 `addNum` 如何维持这两个条件：先把新数压进 `left`（大顶堆），再把 `left` 堆顶（左半边最大的）转移到 `right`，相当于给两个堆做了一次「排序归位」，保证条件 2。转移后若 `right` 反而比 `left` 多，就从 `right` 把堆顶（右半边最小的）挪回 `left`，保证条件 1。经过这两步，两个不变量始终成立。

为什么用两个堆而不是每次排序：每次取中位数都重新排序是 O(n log n)。对顶堆把插入控制在 O(log n)、取中位数 O(1)，非常适合「数据源源不断、随时问中位数」的场景。

Python 的 `heapq` 只有小顶堆，所以 `left` 里存相反数来模拟大顶堆。

**代码**（`src/heap/find_median_from_data_stream.py` / `.cpp`）：

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.left = []   # 大顶堆（存相反数），放较小的一半
        self.right = []  # 小顶堆，放较大的一半

    def add_num(self, num):
        heapq.heappush(self.left, -num)
        heapq.heappush(self.right, -heapq.heappop(self.left))
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))

    def find_median(self):
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2
```

```cpp
class MedianFinder {
public:
    void addNum(int num) {
        left_.push(num);
        right_.push(left_.top());
        left_.pop();
        if (right_.size() > left_.size()) {
            left_.push(right_.top());
            right_.pop();
        }
    }

    double findMedian() const {
        if (left_.size() > right_.size()) return left_.top();
        return (left_.top() + right_.top()) / 2.0;
    }

private:
    std::priority_queue<int> left_;                                  // 大顶堆，较小的一半
    std::priority_queue<int, std::vector<int>, std::greater<int>> right_;  // 小顶堆，较大的一半
};
```

- **复杂度**：`addNum` 时间 O(log n)，`findMedian` 时间 O(1)，空间 O(n)。
- **易错点**：两个堆的**方向别搞反**——左边（较小的一半）要能 O(1) 取最大值，所以是大顶堆；右边（较大的一半）要能 O(1) 取最小值，所以是小顶堆。Python 里 `left` 存的是负值，读堆顶和求平均时都要还原符号；偶数时求和后除以 `2` 用浮点（C++ 写 `/ 2.0`），否则整数除法会丢精度；每次插入都要先转移再平衡，两步缺一不可。
- **相似题**：703. 数据流中的第 K 大元素（对顶堆的简化版，只要一边，见上）；480. 滑动窗口中位数（对顶堆再加一个「延迟删除」处理滑出元素）；面试常把本题当「两个堆维护动态中位数」的模板。

---

## 模式三：多路归并

**适用信号**：有**多条各自有序的序列**（多行、多条链表、两个数组），
要求「整体第 K 小」或「整体最小的 K 个」。核心动作：把所有序列的**当前头**放进堆，
堆顶就是全局极值；弹出一个后，把它所在序列的下一个补进堆。
求最小用小顶堆、求最大用大顶堆；堆里除了「值」还要带**来源下标**，才能知道下一个是谁。

### 23. 合并 K 个升序链表（困难）

**题目**：给定一个链表数组，每个链表都已经按升序排列。请将所有链表合并成一个升序链表并返回。

**思路（小顶堆做多路归并）**：
这是「归并两个有序链表」的推广：当有 k 条链时，暴力做法是两两合并，每次都要重新比较 k 个头结点，复杂度会退化。用小顶堆把「当前所有链表的头结点」放进一个池子，堆顶就是全局最小的结点：每次弹出堆顶，接到结果链表尾部；然后把这个结点的下一个再压回堆。堆里始终只有至多 k 个候选，取最小值只要 O(log k)。

为什么堆里要存一个额外下标：Python/C++ 的堆在比较元素时，如果值相等会继续比较后面的字段；如果直接放链表结点，相等时就会去比较结点对象，报错或行为未定义。所以打包成 `(结点值, 唯一序号, 结点)`：值相等时用递增序号区分，绝对不会比到结点本身。

另一种做法是分治：把 k 条链两两配对合并，共 log k 轮，每轮 O(n)。两者都是 O(n log k)，堆解更直观，分治解在分治篇里再讲。

**代码**（`src/heap/merge_k_sorted_lists.py` / `.cpp`）：

```python
import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_k_lists(lists):
    heap = []
    counter = 0
    for head in lists:
        if head is not None:
            heapq.heappush(heap, (head.val, counter, head))
            counter += 1

    dummy = ListNode()
    tail = dummy
    while heap:
        _, _, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    tail.next = None
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeKLists(std::vector<ListNode *> lists) {
    using Item = std::tuple<int, int, ListNode *>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    int counter = 0;
    for (ListNode *head : lists) {
        if (head) minHeap.push({head->val, counter++, head});
    }

    ListNode dummy;
    ListNode *tail = &dummy;
    while (!minHeap.empty()) {
        auto [val, idx, node] = minHeap.top();
        minHeap.pop();
        tail->next = node;
        tail = node;
        if (node->next) minHeap.push({node->next->val, counter++, node->next});
    }
    tail->next = nullptr;
    return dummy.next;
}
```

- **复杂度**：时间 O(n log k)（n 为结点总数，k 为链表条数），空间 O(k)。
- **易错点**：堆里必须带**唯一序号**——只放 `(值, 结点)` 时，值相等会比较结点本身导致运行错误；空链表要先跳过，不能压入 `None`；记得把结果链表末尾的 `next` 置空，否则会拖上原链表的残留尾巴；用虚拟头结点 `dummy` 能省掉「结果为空」的边界讨论，这是链表题的通用技巧。
- **相似题**：378. 有序矩阵中第 K 小的元素（把「k 条链表」换成「n 行」）；373. 查找和最小的 K 对数字（把「k 条链表」换成「两个数组的配对」）；21. 合并两个有序链表（k = 2 的退化版，链表篇讲过）。

### 378. 有序矩阵中第 K 小的元素（中等）

**题目**：给你一个 `n x n` 矩阵，每行、每列都按升序排列。返回矩阵中第 `k` 小的元素（按排序顺序，不是第 `k` 个不同元素）。

**思路（多路归并：把每一行看成一条有序链，用堆取第 k 个）**：
每行都是升序的，所以「整个矩阵的第 k 小」等价于把 n 条有序行归并后取第 k 个。做法和 23 题合并 K 个有序链表一模一样：先把每一行的第一个元素 `(值, 行, 列)` 压入小顶堆，堆顶就是当前全局最小。每弹出一个，就把它所在行的下一个元素压回堆。弹出 k-1 次后，堆顶就是第 k 小。

为什么不用把整个矩阵读出来排序：矩阵有 n² 个元素，全部排序要 O(n² log n)。多路归并只需要维护 n 个候选（每行一个），每次 O(log n)，总共 O(k log n)，当 k 远小于 n² 时快得多，也更省内存。

另一种做法是二分答案：对值域二分，统计不超过 `mid` 的元素个数，也能做到 O(n log(max-min))，思路在二分篇里讲，这里详展更贴合堆主题的归并解。

**代码**（`src/heap/kth_smallest_element_in_a_sorted_matrix.py` / `.cpp`）：

```python
import heapq


def kth_smallest(matrix, k):
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)

    for _ in range(k - 1):
        value, i, j = heapq.heappop(heap)
        if j + 1 < len(matrix[i]):
            heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))

    return heap[0][0]
```

```cpp
int kthSmallest(const std::vector<std::vector<int>> &matrix, int k) {
    int n = matrix.size();
    using Item = std::tuple<int, int, int>;  // (值, 行, 列)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    for (int i = 0; i < n; ++i) minHeap.push({matrix[i][0], i, 0});

    for (int step = 0; step < k - 1; ++step) {
        auto [value, i, j] = minHeap.top();
        minHeap.pop();
        if (j + 1 < (int)matrix[i].size()) {
            minHeap.push({matrix[i][j + 1], i, j + 1});
        }
    }
    return std::get<0>(minHeap.top());
}
```

- **复杂度**：时间 O(k log n)，空间 O(n)。
- **易错点**：每个元素是「弹出后才补下一个」，所以循环次数是 `k - 1` 次，最后直接取堆顶；别在弹第 k 次之后才取堆顶，那样会少算或多算；列越界判断 `j + 1 < len(matrix[i])`（注意取的是第 i 行的长度）；C++ 用 `tuple` 的 `std::get<0>` 取值（或结构化绑定）。
- **相似题**：23. 合并 K 个升序链表（同一多路归并模板）；373. 查找和最小的 K 对数字（同为「有序行列取最小」）；视频/二分篇会用「二分答案 + 计数」给出另一解。

### 373. 查找和最小的 K 对数字（中等）

**题目**：给定两个以升序排列的整数数组 `nums1` 和 `nums2`，以及整数 `k`。定义一对数字 `(u, v)`：第一个来自 `nums1`，第二个来自 `nums2`。返回和最小的 k 对数字。

**思路（小顶堆 + 单调剪枝，多路归并的又一实例）**：
两个数组都升序，所以对固定的 i，随着 j 增大 `nums1[i] + nums2[j]` 单调不减。于是可以把「每一行 i」看成一个升序序列，问题变成从 k 个有序序列里取最小的 k 个——正是多路归并。

做法：初始化时，把每个 `nums1[i]`（i 只取前 k 个就够）与 `nums2[0]` 组成 `(和, i, 0)` 压入小顶堆。每弹出一个 `(i, j)`，就把它「这一行的下一个」 `(i, j+1)` 压回堆。弹出 k 次即得答案。

为什么 i 只需枚举前 k 个：第 j 列同理，答案里的下标不会超过 `k-1`——因为即使最差的情况，也只需要每行/每列贡献少数几个。这个剪枝让建堆规模从 O(m) 降到 O(k)。

**代码**（`src/heap/find_k_pairs_with_smallest_sums.py` / `.cpp`）：

```python
import heapq


def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k <= 0:
        return []

    heap = []
    for i in range(min(len(nums1), k)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))

    result = []
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return result
```

```cpp
std::vector<std::vector<int>> kSmallestPairs(const std::vector<int> &nums1,
                                             const std::vector<int> &nums2, int k) {
    std::vector<std::vector<int>> result;
    if (nums1.empty() || nums2.empty() || k <= 0) return result;

    using Item = std::tuple<int, int, int>;  // (和, i, j)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    int rows = std::min((int)nums1.size(), k);
    for (int i = 0; i < rows; ++i) {
        minHeap.push({nums1[i] + nums2[0], i, 0});
    }

    while (!minHeap.empty() && (int)result.size() < k) {
        auto [sum, i, j] = minHeap.top();
        minHeap.pop();
        result.push_back({nums1[i], nums2[j]});
        if (j + 1 < (int)nums2.size()) {
            minHeap.push({nums1[i] + nums2[j + 1], i, j + 1});
        }
    }
    return result;
}
```

- **复杂度**：时间 O(k log k)，空间 O(k)。
- **易错点**：堆里存的是 `(和, i, j)`，弹出后要用 `i, j` 去原数组取值，别把「和」当成元素输出；循环条件要同时判断堆非空和结果不足 k，否则数组很短时会多弹；C++ 建堆行数取 `min(nums1.size(), k)`；空数组要提前返回。
- **相似题**：378. 有序矩阵中第 K 小的元素（「二维有序取最小」的同型题）；23. 合并 K 个升序链表（把「和」换成「结点值」）；题目是「多路归并求前 K 小」的最纯粹形式。

---

## 模式四：大顶堆：每次取当前最大值

**适用信号**：模拟过程中**反复需要当前最大（或最小）的元素**，且元素会被「取走后
消耗/合并/修改」。典型是「每次挑两个最大的做运算」。这时堆就是天然的模拟引擎：
取最大值 O(log n)、放回新值 O(log n)。

### 1046. 最后一块石头的重量（简单）

**题目**：有一堆石头，每块石头的重量是正整数。每次选出两块最重的石头，让它们相撞：若重量相等，两块都碎掉；否则较重的一块剩下，新重量为两者之差。重复到至多剩一块石头，返回它的重量（没有石头则返回 0）。

**思路（用大顶堆每次取两个最大值）**：
题目要求「每次取最重的两块」，这正是优先队列最擅长的：把所有石头放进一个大顶堆，堆顶就是当前最重的。循环：弹出两块最重的 `a`、`b`（`a >= b`），若 `a != b`，把差值 `a - b` 压回堆。直到堆里不足两块，返回剩下那块的重量（空则 0）。

为什么用大顶堆而不是每次排序：每次模拟都要取当前最大值，若每轮重新排序会退化成 O(n² log n)。堆能 O(log n) 地取出最大值、O(log n) 地插回新值，总共最多做 n 次「取出两块、插回一块」，整体 O(n log n)。

Python 的 `heapq` 是小顶堆，所以把石头取负号再放进去，取出来时再取负号还原。C++ 的 `priority_queue` 默认就是大顶堆，直接可用。

**代码**（`src/heap/last_stone_weight.py` / `.cpp`）：

```python
import heapq


def last_stone_weight(stones):
    max_heap = [-s for s in stones]
    heapq.heapify(max_heap)
    while len(max_heap) > 1:
        first = -heapq.heappop(max_heap)
        second = -heapq.heappop(max_heap)
        if first != second:
            heapq.heappush(max_heap, -(first - second))
    return -max_heap[0] if max_heap else 0
```

```cpp
int lastStoneWeight(std::vector<int> stones) {
    std::priority_queue<int> maxHeap(stones.begin(), stones.end());
    while (maxHeap.size() > 1) {
        int first = maxHeap.top();
        maxHeap.pop();
        int second = maxHeap.top();
        maxHeap.pop();
        if (first != second) maxHeap.push(first - second);
    }
    return maxHeap.empty() ? 0 : maxHeap.top();
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：Python 里**符号取两次**（存入时取负、取出时再取负），漏掉一次会得到完全错误的顺序；相减的顺序是「大减小」，因为 `first >= second`；循环条件是 `> 1`（不足两块就停），最后要处理堆为空（`[]` 或全碎）返回 0；C++ 的 `priority_queue` 可以直接用向量迭代器区间构造。
- **相似题**：215. 数组中的第 K 个最大元素（「从一堆里反复取最大」的另一应用）；295. 数据流的中位数（同时需要最大和最小，于是用两个堆）；本质是「堆做事件模拟」的入门题。

---

## 模式五：自定义优先级

**适用信号**：Top-K 的排序依据不是单调的整数，而是**自定义的规则**（距离、字典序、
多关键字组合）。做法是先算出每个候选的排序键，再把它和要输出的内容一起打包进堆。
关键是想清楚「谁该排在前面」，以及 Python/C++ 的堆方向如何对应。

### 973. 最接近原点的 K 个点（中等）

**题目**：给定一个点数组 `points` 和一个整数 `k`，返回距离原点 `(0, 0)` 最近的 k 个点。两点之间的距离用欧几里得距离；答案顺序任意。

**思路（维护一个大小为 k 的大顶堆，留下最近的 k 个）**：
这是 215 的「镜像版」：215 求第 K 大，用大小为 k 的**小顶堆**当淘汰线；本题求第 K 小（最近），就用大小为 k 的**大顶堆**——堆顶是当前保留集合里最远的那个，正好当淘汰线：新点比堆顶近就挤掉堆顶，否则不留。遍历完，堆里就是最近的 k 个点。

为什么比较距离的平方而不是开根号：距离是 `sqrt(x² + y²)`，单调递增，比较大小等价于比较 `x² + y²`。不开根既省时间又避免浮点误差，这是个常用小技巧。

为什么 Python 里存「负距离的平方」：`heapq` 是小顶堆，要模拟大顶堆只能存相反数；弹出时堆顶对应 `-dist` 最小，也就是 `dist` 最大，即最远的点。

**代码**（`src/heap/k_closest_points_to_origin.py` / `.cpp`）：

```python
import heapq


def k_closest(points, k):
    max_heap = []
    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(max_heap, (-dist, x, y))
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    return [[x, y] for _, x, y in max_heap]
```

```cpp
std::vector<std::vector<int>> kClosest(std::vector<std::vector<int>> &points, int k) {
    // 默认大顶堆，堆顶是当前保留点里距离平方最大的（最远），正好当淘汰线。
    std::priority_queue<std::pair<int, std::pair<int, int>>> maxHeap;
    for (const auto &point : points) {
        int x = point[0], y = point[1];
        maxHeap.push({x * x + y * y, {x, y}});
        if ((int)maxHeap.size() > k) maxHeap.pop();
    }

    std::vector<std::vector<int>> result;
    while (!maxHeap.empty()) {
        auto [dist, coord] = maxHeap.top();
        maxHeap.pop();
        result.push_back({coord.first, coord.second});
    }
    return result;
}
```

- **复杂度**：时间 O(n log k)，空间 O(k)。
- **易错点**：求**最小**要用**大顶堆**——和 215 求最大用小顶堆正好相反，这里最容易记反；用距离的平方比较，别开根号；Python 存 `(-dist, x, y)`，取元素时跳过第一项拿 `x, y`；答案顺序任意，测试时先排序再比。
- **相似题**：215. 数组中的第 K 个最大元素（「求最大用小顶堆」，本题是它严格的对偶）；692. 前 K 个高频单词（另一类自定义优先级，见下）；面试中「距离最近」「价值最高」的 Top-K 都套这一模板。

### 692. 前 K 个高频单词（中等）

**题目**：给一个单词列表 `words` 和整数 `k`，返回出现频率前 `k` 高的单词。返回顺序按频率从高到低；频率相同时，按字典序从小到大排列。

**思路（哈希计数 + 带自定义优先级的堆）**：
先用哈希表统计每个单词的出现次数，得到一批 `(单词, 频率)`。排序规则是「频率高的在前；频率相同则字典序小的在前」。把每个单词打包成 `(-频率, 单词)` 放进小顶堆，然后依次弹出 k 个：

- 按 `-频率` 从小到大，正好是频率从大到小；
- 频率相同时按单词从小到大，正好是字典序。

所以堆的弹出顺序恰好就是题目要求的顺序，弹 k 次即得答案。

为什么用 `-频率` 而不是频率：Python 的 `heapq` 是小顶堆，弹出的是最小元素。频率越大越靠前，就取相反数让「大频率」对应「更小的键」，弹出来自然是从高频到低频。

C++ 的 `priority_queue` 可以直接传一个比较器，把「频率高优先、同频字典序小优先」写成偏序关系，堆顶就是最该输出的单词。

另一种做法是「大小为 k 的小顶堆淘汰线」：维护 k 个最优，堆顶放「最差」的那个（频率最低，同频字典序最大）。Python 里要自定义比较器稍麻烦，本篇直接用全量堆 + 弹 k 次，代码更短且同样高效。

**代码**（`src/heap/top_k_frequent_words.py` / `.cpp`）：

```python
import heapq


def top_k_frequent_words(words, k):
    count = {}
    for word in words:
        count[word] = count.get(word, 0) + 1

    heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(heap)

    return [heapq.heappop(heap)[1] for _ in range(k)]
```

```cpp
std::vector<std::string> topKFrequent(std::vector<std::string> &words, int k) {
    std::unordered_map<std::string, int> count;
    for (const std::string &word : words) ++count[word];

    using P = std::pair<std::string, int>;  // (单词, 频率)
    auto better = [](const P &a, const P &b) {
        if (a.second != b.second) return a.second < b.second;  // 频率低的优先级低
        return a.first > b.first;                              // 同频字典序大的优先级低
    };
    std::priority_queue<P, std::vector<P>, decltype(better)> maxHeap(better);
    for (const auto &[word, freq] : count) maxHeap.push({word, freq});

    std::vector<std::string> result;
    for (int i = 0; i < k && !maxHeap.empty(); ++i) {
        result.push_back(maxHeap.top().first);
        maxHeap.pop();
    }
    return result;
}
```

- **复杂度**：时间 O(m + k log m)（m 为不同单词个数），空间 O(m)。
- **易错点**：Python 元组是 `(-频率, 单词)`，负号只加在频率上，**不要给单词加负号**（字符串不支持）；要的是「同频字典序小在前」，而 `heapq` 弹最小，所以负频率配上正序单词恰好满足，别多加一层反转；C++ 比较器里两个条件的**方向**：频率低的、字典序大的，才是「差」的、该排在堆顶后面的；返回的是 `second`/`.first` 这类正确字段（Python 中单词在索引 1，C++ 中单词在 `.first`）。
- **相似题**：347. 前 K 个高频元素（不带字典序的简化版，见「模式一」）；973. 最接近原点的 K 个点（另一种自定义排序键）；451. 根据字符出现频率排序（自定义优先级排序的另一应用）。

---

## 规律总结

1. **Top-K 的通用模板**：维护一个**大小为 k 的堆**，遍历数据，先 `push`，若堆大小超过 k 就 `pop` 一次；最后堆里就是「最靠前的 k 个」，堆顶是第 k 名。215 按数值排、347 按频率排、973 按距离排，只是入堆的「依据」不同，骨架完全一样。

2. **求最大用最小堆，求最小用最大堆**。这条初看反直觉，道理却简单：堆顶是「保留集合里最差的」，正好充当淘汰线，每次只淘汰最差的那个。215 求第 K 大用**小顶堆**（淘汰线是最小值），973 求第 K 小（最近）用**大顶堆**（淘汰线是最大值），二者互为镜像。Python 的 `heapq` 天然是小顶堆；C++ 的 `priority_queue` 默认大顶堆，要小顶堆必须传 `std::greater<T>`，这是最容易写反的地方。

3. **在线 vs 离线，决定堆的用法**。离线数据可以全部拿到后再建大小为 k 的堆（215、347）；在线数据只能在每个新元素到来时维护堆，堆始终固定在 k 个（703），或两堆各半（295）。「数据流」题的共同特征是：存不下全部数据，只能靠滚动的小堆维护答案。

4. **多路归并的骨架是「堆里放各序列的当前头」**。23 放各链表的头结点，378 放各行的首元素，373 放每个 `i` 对应 `(i, 0)` 的配对。每弹出一个，就把同一序列的**下一个**补进堆；求最小用小顶堆，求最大用大顶堆。堆里除了值一定要带**来源下标**，否则不知道下一个是谁；比较对象可能相等时，还要带**唯一序号**避免比到对象本身。

5. **需要比较多个维度时，把依据和结果打包成元组/pair**。347 要按频率排却输出元素，存 `(频率, 元素)`；373 要按和排却输出数对，存 `(和, i, j)`；692 要按「频率降序、字典序升序」排，存 `(-频率, 单词)`。规则是「先放排序依据，再放要输出的东西」，取结果时别拿错字段。C++ 里多关键字排序用自定义比较器，注意把「谁该排前面」翻译成「谁优先级更低」。

6. **什么时候该用堆，而不是排序**：当只需要前 k 个、或需要反复取当前最值时。前者用大小为 k 的堆把复杂度从 O(n log n) 降到 O(n log k)；后者（1046、295）根本没法一次排好序，只能靠堆滚动维护。堆不擅长的事是「查任意第几小」和「区间查询」，那需要别的结构。

7. **距离、平方、和这类「单调的量」可以直接比，不必算出真实值**。973 比较距离的平方而非开根号，既快又避免浮点误差。凡是比较只用到大小关系、而实际值不参与后续计算时，都可以用这个技巧。

8. **`docs` 与 `src` 必须逐字一致**。题解里的代码与 `code/leetcode/src/heap/` 下的实现保持完全一致，以经过自测的 `src` 为准，文档只做粘贴，避免两边脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
