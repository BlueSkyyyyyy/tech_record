# 堆与 Top-K

堆（heap）是一种能**快速取出最值**的数据结构：插入、删除、取最值都是 O(log n)。
它最常见的用途，是解决一类叫 **Top-K** 的问题——「最大的 k 个」「频率最高的 k 个」「第 k 大」。
这类问题的共同点是：我们并不需要把全部数据排好序，只关心**最靠前的一小撮**。

本篇只讲一个核心招式：**用大小为 k 的小顶堆当「淘汰线」**。
它看起来反直觉——求最大的 k 个，为什么用小顶堆（堆顶最小）？答案正是：堆顶就是那条线，
每次只淘汰当前最小、留下最有资格的 k 个。215 和 347 是同一个套路，只是「排序依据」从数值换成了频率。

> 说明：Python 的 `heapq` 只提供**小顶堆**；C++ 的 `priority_queue` 默认是大顶堆，
> 想用小顶堆要额外传比较器 `std::greater<T>`。这个差别是本题型最容易踩的坑。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 小顶堆选第 K 大 | 215. 数组中的第 K 个最大元素 | 中等 |
| 小顶堆选前 K 高频 | 347. 前 K 个高频元素 | 中等 |

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
- **相似题**：347. 前 K 个高频元素（同一模板，排序依据换成频率，见下）；703. 数据流中的第 K 大元素（把一次性的数组换成持续到来的数据流，维护同样的小顶堆）；1046. 最后一块石头的重量（每次取两个最大值，用大顶堆，见 `heap` 续篇）；本题和 215 的快速选择解也常与分治篇的 `215` 交叉讲解。

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
- **相似题**：692. 前 K 个高频单词（同题加一个「同频按字典序」的比较规则，见 `heap` 续篇）；215. 数组中的第 K 个最大元素（同一「小顶堆 + 限制 k」模板，见上）；451. 根据字符出现频率排序（统计频率后的另一种应用）。

---

## 规律总结

1. **Top-K 的通用模板**：维护一个**大小为 k 的小顶堆**，遍历数据，先 `push`，若堆大小超过 k 就 `pop` 一次；最后堆里就是「最靠前的 k 个」，堆顶是第 k 名。215 按数值排、347 按频率排，只是入堆的「依据」不同，骨架完全一样。

2. **求最大用最小堆，求最小用最大堆**。这条初看反直觉，但道理很简单：小顶堆的堆顶是「保留集合里最差的」，正好充当淘汰线，每次只淘汰最差的那个。Python 的 `heapq` 天然是小顶堆；C++ 的 `priority_queue` 默认大顶堆，要小顶堆必须传 `std::greater<T>`，这是最容易写反的地方。

3. **为什么复杂度里 log 的底是 k 而不是 n**：堆的大小被限制在 k，堆高就是 log k。所以当 k 远小于 n 时，O(n log k) 明显优于完整排序的 O(n log n)。面试时能说清「O(n log k) 还是 O(n log n)」往往比写出代码更加分。

4. **需要比较多个维度时，把依据和结果打包成元组**。347 要按频率排序却输出元素，于是存 `(频率, 元素)`；Python 元组、C++ `pair` 都会按第一维、第二维依次比较。记住「先放排序依据，再放要输出的东西」，取结果时别拿错字段。

5. **当数据量很大而 k 很小时，堆解是最省空间的**。它只需要 O(k) 的堆，而不是把所有数据排序所需的 O(n)。「数据流」场景（703、295）尤其如此——数据源源不断，根本存不下全部，只能靠一个小堆滚动维护答案。

6. **`docs` 与 `src` 必须逐字一致**。题解里的代码与 `code/leetcode/src/heap/` 下的实现保持完全一致，以经过自测的 `src` 为准，文档只做粘贴，避免两边脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
