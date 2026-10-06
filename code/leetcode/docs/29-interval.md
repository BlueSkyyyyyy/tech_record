# 区间与扫描线：合并、相交与面积覆盖

「区间」类题目的共同点：数据是一堆 `[左端, 右端]`，问合并、插入、求交、求差、
或者在一堆矩形上算面积。看似种类繁多，其实先动手的永远是同一件事——**排序**，
排序之后问题往往退化成一趟线性扫描。本篇收 10 道题，按五种套路分组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：整理区间列表（合并 / 插入 / 去覆盖） | 228. 汇总区间 / 57. 插入区间 / 1288. 删除被覆盖区间 | 简单 / 中等 / 中等 |
| 模式二：两个有序列表的双指针 | 986. 区间列表的交集 / 1272. 删除区间 | 中等 / 中等 |
| 模式三：动态维护不相交区间集合 | 352. 将数据流变为多个不相交区间 | 困难 |
| 模式四：离线排序 + 扫描线 | 1851. 包含每个查询的最小区间 / 218. 天际线问题 | 困难 / 困难 |
| 模式五：矩形并集与完美覆盖 | 850. 矩形面积 II / 391. 完美矩形 | 困难 / 困难 |

> 一个反复出现的判断：**区间在端点处是闭的还是开的**。求交、相减、扫描线里
> 「离开的楼」何时结算，答案都藏在这一个等号里。本篇会在「规律总结」里专门
> 对照它，这与第 04 篇「前缀和与差分」里差分区间的开闭是同一类细节。

---

## 模式一：整理区间列表

**适用信号**：给一串区间，要求输出「合并后」「插入后」「删除被覆盖后」的规范列表。

**核心动作**：先按左端点排序，再用一个「当前正在维护的区间」向右推进，逐个
判断下一个区间是「接上」还是「另起一段」。

### 228. 汇总区间（简单）

**题目**：给定一个无重复元素的有序整数数组，返回恰好覆盖所有数字的最小有序区间
列表，单个数写成 `"a"`，连续的一段写成 `"a->b"`。

**思路**：

```python
def summary_ranges(nums):
    res = []
    i, n = 0, len(nums)
    while i < n:
        start = nums[i]
        while i + 1 < n and nums[i + 1] == nums[i] + 1:
            i += 1
        if start == nums[i]:
            res.append(str(start))
        else:
            res.append(f"{start}->{nums[i]}")
        i += 1
    return res
```

```cpp
std::vector<std::string> summaryRanges(std::vector<int>& nums) {
    std::vector<std::string> res;
    int n = static_cast<int>(nums.size());
    int i = 0;
    while (i < n) {
        int start = nums[i];
        while (i + 1 < n && nums[i + 1] == nums[i] + 1) {
            ++i;
        }
        if (start == nums[i]) {
            res.push_back(std::to_string(start));
        } else {
            res.push_back(std::to_string(start) + "->" + std::to_string(nums[i]));
        }
        ++i;
    }
    return res;
}
```

**为什么会这么简单**：数组已经有序且无重复，所以「连续」的判定就是相邻差 1。
用 `start` 记住当前段的开头，让 `i` 一路往右走到走不动，`[start, nums[i]]` 就是
一段。写的时候要注意外层循环里 `i += 1` 的位置——内层循环结束时 `i` 停在段的
末尾，必须再走一步才是下一段的开头。

- **复杂度**：时间 O(n)，空间 O(1)（不计返回结果）。
- **易错点**：单个数字时输出 `"a"` 而不是 `"a->a"`；空数组直接返回空列表；
  负数用 `str` / `to_string` 会自带负号，无需特判。
- **相似题**：57. 插入区间（区间列表的新增与合并）；163. 缺失的区间（反过来
  找「没被覆盖的段」）可作为镜像练习。

### 57. 插入区间（中等）

**题目**：给定无重叠且按左端点升序的区间列表，插入一个新区间并合并重叠部分。

**思路**：

```python
def insert(intervals, new_interval):
    res = []
    i, n = 0, len(intervals)
    start, end = new_interval

    while i < n and intervals[i][1] < start:
        res.append(intervals[i])
        i += 1

    while i < n and intervals[i][0] <= end:
        start = min(start, intervals[i][0])
        end = max(end, intervals[i][1])
        i += 1

    res.append([start, end])

    while i < n:
        res.append(intervals[i])
        i += 1

    return res
```

```cpp
std::vector<std::vector<int>> insert(std::vector<std::vector<int>>& intervals,
                                     std::vector<int>& newInterval) {
    std::vector<std::vector<int>> res;
    int n = static_cast<int>(intervals.size());
    int i = 0;
    int start = newInterval[0], end = newInterval[1];

    while (i < n && intervals[i][1] < start) {
        res.push_back(intervals[i]);
        ++i;
    }
    while (i < n && intervals[i][0] <= end) {
        start = std::min(start, intervals[i][0]);
        end = std::max(end, intervals[i][1]);
        ++i;
    }
    res.push_back({start, end});
    while (i < n) {
        res.push_back(intervals[i]);
        ++i;
    }
    return res;
}
```

**为什么分三段扫描**：原列表已经有序，插入只会影响一个连续的「重叠块」。
第一段是新区间左边、右端点小于 `start` 的（完全不相交，原样保留）；第二段是
左端点不超过 `end` 的（与新区间重叠，每遇到一个就把 `[start, end]` 扩张到它的
两端，等于把整块吸进来）；剩下的第三段同样不相交。最后把扩张后的区间插在
第一、三段中间。三个 `while` 里 `i` 是共享的，天然接力。

- **复杂度**：时间 O(n)，空间 O(1)（不计返回结果）。
- **易错点**：判断「完全在左边」用的是 `intervals[i][1] < start`（严格小于），
  判断「重叠」用的是 `intervals[i][0] <= end`（小于等于）——一个开一个闭，写反
  会把「恰好接上」的区间漏掉或多算；扩张时两端都要取 `min`/`max`。
- **相似题**：56. 合并区间（区间列表的纯合并，见第 01 篇）；435. 无重叠区间 /
  452. 引爆气球（区间调度，见第 14 篇贪心）。

### 1288. 删除被覆盖区间（中等）

**题目**：删除所有被另一个区间完全覆盖的区间，返回剩余区间的数量。

**思路**：

```python
def remove_covered_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: (x[0], -x[1]))
    count = 0
    max_end = -1
    for _, end in intervals:
        if end > max_end:
            count += 1
            max_end = end
    return count
```

```cpp
int removeCoveredIntervals(std::vector<std::vector<int>>& intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  if (a[0] != b[0]) {
                      return a[0] < b[0];
                  }
                  return a[1] > b[1];
              });
    int count = 0;
    int maxEnd = -1;
    for (const auto& iv : intervals) {
        if (iv[1] > maxEnd) {
            ++count;
            maxEnd = iv[1];
        }
    }
    return count;
}
```

**为什么排成「左升右降」**：对当前区间 `[l, r]`，它被覆盖需要一个左端不大于 `l`、
右端不小于 `r` 的区间。按左端升序排后，所有已扫描区间的左端都 `<= l`，覆盖的
唯一悬念就只剩右端：只要之前出现过 `>= r` 的右端，它就被覆盖。于是用一个
`max_end` 记录扫描过的最大右端，`r > max_end` 才保留。左端相同时把右端从大到小
排，能让「更长的那条」先出现，短的自然被覆盖——否则同左端时短的可能先被误当
成「新的最右」。

- **复杂度**：排序 O(n log n)，扫描 O(n)。
- **易错点**：排序键是 `(左升, 右降)` 两段，只按左端排会出错；`max_end` 的初值
  取 `-1` 即可（坐标非负），若坐标可能为负就取更小的哨兵。
- **相似题**：56. 合并区间（同样先排序再逐个决策）；850 / 391（区间覆盖的面积
  版本）。

---

## 模式二：两个有序列表的双指针

**适用信号**：给了两个各自有序、互不相交的区间列表，要求它们的交集或差集。

**核心动作**：两个指针 `i`、`j` 各扫一个列表，每一轮先算出一段结果，再让
「右端点较小的那个」前进——因为它的右端已经用尽，不可能再与后面产生交集。

### 986. 区间列表的交集（中等）

**题目**：给定两个由互不相交且已排序的闭区间组成的列表，返回它们的交集列表。

**思路**：

```python
def interval_intersection(first, second):
    res = []
    i = j = 0
    while i < len(first) and j < len(second):
        lo = max(first[i][0], second[j][0])
        hi = min(first[i][1], second[j][1])
        if lo <= hi:
            res.append([lo, hi])
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return res
```

```cpp
std::vector<std::vector<int>> intervalIntersection(
    std::vector<std::vector<int>>& first, std::vector<std::vector<int>>& second) {
    std::vector<std::vector<int>> res;
    int i = 0, j = 0;
    int m = static_cast<int>(first.size());
    int n = static_cast<int>(second.size());
    while (i < m && j < n) {
        int lo = std::max(first[i][0], second[j][0]);
        int hi = std::min(first[i][1], second[j][1]);
        if (lo <= hi) {
            res.push_back({lo, hi});
        }
        if (first[i][1] < second[j][1]) {
            ++i;
        } else {
            ++j;
        }
    }
    return res;
}
```

**为什么指针只动一个**：两个区间 `[l1, r1]`、`[l2, r2]` 的交集是
`[max(l1, l2), min(r1, r2)]`，当且仅当左端不超过右端时存在。算完之后，
右端点较小的那个区间已经「用完」——它的右端比对方小，和对方后面的任何区间
都不可能再重叠；而右端点较大的那个还可能与下一对相交，所以保留。两个指针各自
单调前进，总移动次数不超过两表长度之和。

- **复杂度**：时间 O(m + n)，空间 O(1)（不计返回结果）。
- **易错点**：交集判定要用 `lo <= hi`（闭区间，端点相接也算一个长度为 0 或 1 的
  区间），写成 `<` 会漏掉 `[5,5]` 这种点交集；每次循环只能有一个指针前进，
  否则会跳过答案。
- **相似题**：1272. 删除区间（同一套双列表思维，但输出是差集）。

### 1272. 删除区间（中等）

**题目**：给定有序且互不重叠的区间列表，以及要删掉的区间 `[lo, hi]`，返回剩余
区间。

**思路**：

```python
def remove_interval(intervals, to_be_removed):
    lo, hi = to_be_removed
    res = []
    for a, b in intervals:
        if b <= lo or a >= hi:
            res.append([a, b])
        else:
            if a < lo:
                res.append([a, lo])
            if b > hi:
                res.append([hi, b])
    return res
```

```cpp
std::vector<std::vector<int>> removeInterval(
    std::vector<std::vector<int>>& intervals, std::vector<int>& toBeRemoved) {
    int lo = toBeRemoved[0], hi = toBeRemoved[1];
    std::vector<std::vector<int>> res;
    for (const auto& iv : intervals) {
        int a = iv[0], b = iv[1];
        if (b <= lo || a >= hi) {
            res.push_back({a, b});
        } else {
            if (a < lo) {
                res.push_back({a, lo});
            }
            if (b > hi) {
                res.push_back({hi, b});
            }
        }
    }
    return res;
}
```

**为什么用两个 `if` 处理残段**：一个区间和删除区间相交后，剩下的部分最多是
「左边一截」和「右边一截」。`a < lo` 说明左端伸出了删除区间的左边，保留
`[a, lo]`；`b > hi` 说明右端伸到了右边，保留 `[hi, b]`。当删除区间完全盖住它时
两个条件都不成立，自动什么都不留；当删除区间被它完全包住时两个条件都成立，
正好切成两段。用一个 `else` 分支统一覆盖所有相交情形，比分类枚举更不容易漏。

- **复杂度**：时间 O(n)，空间 O(1)（不计返回结果）。
- **易错点**：不相交判定 `b <= lo or a >= hi` 必须用「小于等于」「大于等于」，
  因为端点刚好等于删除边界时其实没有被删到；两个残段判断相互独立，不能写成
  `elif`。
- **相似题**：986. 区间列表的交集（求交）；57. 插入区间（求并）。

---

## 模式三：动态维护不相交区间集合

**适用信号**：数据一个个到来，每次插入后都要求当前合并好的区间列表。

**核心动作**：维护一个**有序**的区间列表，新数字只可能影响它左右的「邻居」，
用二分定位后局部合并即可。

### 352. 将数据流变为多个不相交区间（困难）

**题目**：实现 `addNum(val)` 和 `getIntervals()`，前者加入一个数，后者返回当前
所有数字合并后的不相交区间列表。

**思路**：

```python
import bisect


class SummaryRanges:
    def __init__(self):
        self.intervals = []

    def addNum(self, value):
        intervals = self.intervals
        i = bisect.bisect_left(intervals, [value])

        if i > 0 and intervals[i - 1][1] >= value:
            return
        if i < len(intervals) and intervals[i][0] == value:
            return

        if i > 0 and intervals[i - 1][1] == value - 1:
            intervals[i - 1][1] = value
            if i < len(intervals) and intervals[i][0] == value + 1:
                intervals[i - 1][1] = intervals[i][1]
                intervals.pop(i)
            return

        if i < len(intervals) and intervals[i][0] == value + 1:
            intervals[i][0] = value
            return

        intervals.insert(i, [value, value])

    def getIntervals(self):
        return [list(iv) for iv in self.intervals]
```

```cpp
class SummaryRanges {
public:
    void addNum(int value) {
        auto it = std::lower_bound(
            intervals.begin(), intervals.end(), value,
            [](const std::vector<int>& a, int v) { return a[0] < v; });

        if (it != intervals.begin() && (*(it - 1))[1] >= value) {
            return;
        }
        if (it != intervals.end() && (*it)[0] == value) {
            return;
        }

        bool leftAdj =
            (it != intervals.begin() && (*(it - 1))[1] == value - 1);
        bool rightAdj =
            (it != intervals.end() && (*it)[0] == value + 1);

        if (leftAdj && rightAdj) {
            (*(it - 1))[1] = (*it)[1];
            intervals.erase(it);
        } else if (leftAdj) {
            (*(it - 1))[1] = value;
        } else if (rightAdj) {
            (*it)[0] = value;
        } else {
            intervals.insert(it, {value, value});
        }
    }

    std::vector<std::vector<int>> getIntervals() { return intervals; }

private:
    std::vector<std::vector<int>> intervals;
};
```

**为什么只处理左右邻居**：区间列表始终有序且互不相交，加入 `value` 后，它要么
落进某个已有区间（无需操作），要么紧贴左边区间、紧贴右边区间、或独立成段。
用二分找到「第一个左端 >= value」的位置 `i`，`i-1` 和 `i` 就是左右最近的两个
区间，所有情况都能在读这两个邻居后决定。若同时紧贴左右，就把它俩合成一个——
这说明一个新区间最多只能「吃掉」两个邻居，均摊成本很低。

- **复杂度**：`addNum` 二分 O(log n)，列表插入/删除最坏 O(n)（n 为区间数）；
  `getIntervals` O(n)。
- **易错点**：先判「已被覆盖」再判合并，否则重复数字会被当成新段；合并左右
  邻居时先改左邻居的右端、再删右邻居，顺序写反会丢信息；返回快照而不是内部
  引用，防止调用方改动。
- **相似题**：57. 插入区间（一次性版本）；146. LRU / 380. 随机集合（都是
  「组合基础结构 + 局部 O(1) 维护」，见第 19 篇）。

---

## 模式四：离线排序 + 扫描线

**适用信号**：一堆区间配上一堆查询，要求对每个查询回答「哪个区间最优」；
或者求所有矩形轮廓的并集。共同点是**把查询/事件也排序，按坐标从左到右扫**。

**核心动作**：把所有「事件」（查询、区间开始、区间结束）按坐标排序，用指针 +
堆维护「当前时刻仍然有效的候选」，保证每个对象只进出一次。

### 1851. 包含每个查询的最小区间（困难）

**题目**：对每个查询 `q`，求包含它的区间中最短的长度；不存在返回 -1。

**思路**：

```python
import heapq


def min_interval(intervals, queries):
    intervals = sorted(intervals)
    order = sorted(range(len(queries)), key=lambda i: queries[i])
    res = [-1] * len(queries)

    heap = []
    j = 0
    n = len(intervals)
    for i in order:
        q = queries[i]
        while j < n and intervals[j][0] <= q:
            left, right = intervals[j]
            heapq.heappush(heap, (right - left + 1, right))
            j += 1
        while heap and heap[0][1] < q:
            heapq.heappop(heap)
        if heap:
            res[i] = heap[0][0]
    return res
```

```cpp
std::vector<int> minInterval(std::vector<std::vector<int>>& intervals,
                             std::vector<int>& queries) {
    std::sort(intervals.begin(), intervals.end());
    int q = static_cast<int>(queries.size());
    std::vector<int> order(q);
    for (int i = 0; i < q; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return queries[a] < queries[b]; });

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                        std::greater<std::pair<int, int>>>
        heap;
    std::vector<int> res(q, -1);
    int n = static_cast<int>(intervals.size());
    int j = 0;
    for (int idx : order) {
        int x = queries[idx];
        while (j < n && intervals[j][0] <= x) {
            heap.push({intervals[j][1] - intervals[j][0] + 1, intervals[j][1]});
            ++j;
        }
        while (!heap.empty() && heap.top().second < x) {
            heap.pop();
        }
        if (!heap.empty()) {
            res[idx] = heap.top().first;
        }
    }
    return res;
}
```

**为什么离线排序**：包含 `q` 的区间要求 `li <= q <= ri`。查询按 `q` 升序处理后，
「左端不超过当前 q」的区间集合只增不减，于是可以用一个指针 `j` 一次把它们全部
压进堆（键为长度）；同时用「右端 < q」做惰性删除，弹掉已经覆盖不到更大查询的
区间。弹完过期项后堆顶就是最短区间。每个区间只入堆一次、出堆一次，避免了
每个查询都重扫。

- **复杂度**：排序 O(n log n + q log q)，扫描 O((n + q) log n)；空间 O(n + q)。
- **易错点**：答案要按**原始下标**回填（先把下标排序、下标答案）；堆里存的是
  `(长度, 右端)`，惰性删除时只看右端 `< q`；某查询没有可用区间时保持 -1。
- **相似题**：218. 天际线问题（同样的事件排序 + 堆）；253. 会议室 II（求
  同时进行的会议数，是扫描线的经典入门题）。

### 218. 天际线问题（困难）

**题目**：给定每个建筑的 `[左 x, 右 x, 高]`，返回天际线的关键点 `[x, 高度]`。

**思路**：

```python
import heapq


def get_skyline(buildings):
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))  # 进入：高度取负，便于统一排序
        events.append((right, height, right))  # 离开：高度为正
    events.sort()

    res = []
    heap = [(0, float("inf"))]  # (-高度, 右端点)，地面常驻
    for x, signed_height, right in events:
        if signed_height < 0:
            heapq.heappush(heap, (signed_height, right))
        while heap[0][1] <= x:
            heapq.heappop(heap)
        cur = -heap[0][0]
        if not res or res[-1][1] != cur:
            res.append([x, cur])
    return res
```

```cpp
std::vector<std::vector<int>> getSkyline(std::vector<std::vector<int>>& buildings) {
    std::vector<std::array<int, 3>> events;
    for (const auto& b : buildings) {
        events.push_back({b[0], -b[2], b[1]});  // 进入
        events.push_back({b[1], b[2], b[1]});   // 离开
    }
    std::sort(events.begin(), events.end());

    std::priority_queue<std::pair<int, int>> heap;  // (高度, 右端点)
    heap.push({0, INT_MAX});                         // 地面常驻
    std::vector<std::vector<int>> res;
    for (const auto& e : events) {
        int x = e[0], h = e[1], r = e[2];
        if (h < 0) {
            heap.push({-h, r});
        }
        while (heap.top().second <= x) {
            heap.pop();
        }
        int cur = heap.top().first;
        if (res.empty() || res.back()[1] != cur) {
            res.push_back({x, cur});
        }
    }
    return res;
}
```

**为什么把「进入/离开」编码进排序**：每栋楼只在左端「进入」、右端「离开」。
把它们做成两个事件按 x 排序，同一 x 上要让「进入」排在「离开」前面（否则会在
新楼加进来之前误报一次高度下降）——Python 用 `-height` 与 `+height` 的正负、
C++ 用事件数组的 `{x, -h, r}` / `{x, h, r}` 都达到了这个效果。扫描时用一个
大顶堆维护当前仍覆盖的楼高，堆顶若非当前最高，而是已经离开的楼，就用
「右端点 <= 当前 x」把它弹出（惰性删除）。堆顶高度发生变化的那一刻，就是天际线
的一个关键点。

- **复杂度**：事件排序 O(n log n)，每个事件 O(log n)；空间 O(n)。
- **易错点**：同一 x 上「进入」必须先于「离开」；地面用一个高度 0、右端无穷大的
  哨兵常驻，才能正确回落；只有高度**变化**时才记录点，避免输出一堆同高度的
  冗余点；相等高度相邻也要合并。
- **相似题**：1851（事件 + 堆的同一骨架）；850. 矩形面积 II（同样从左往右扫
  x，只是最终求的是面积而非轮廓）。

---

## 模式五：矩形并集与完美覆盖

**适用信号**：一堆轴对齐矩形，求覆盖总面积，或判断能否恰好拼成一个大矩形。

**核心动作**：把 x 方向离散化，逐条竖带统计被覆盖的高度；或者用「面积 + 角点
出现次数」两个不变量直接判定。

### 850. 矩形面积 II（困难）

**题目**：求一组轴对齐矩形覆盖的总面积（重叠只算一次），结果对 `1e9 + 7` 取模。

**思路**：

```python
def rectangle_area(rectangles):
    MOD = 10**9 + 7
    xs = sorted({x for x1, _, x2, _ in rectangles for x in (x1, x2)})
    area = 0
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        spans = []
        for x1, y1, x2, y2 in rectangles:
            if x1 <= xa and xb <= x2:
                spans.append((y1, y2))
        spans.sort()

        covered = 0
        cur_lo = cur_hi = None
        for y1, y2 in spans:
            if cur_hi is None:
                cur_lo, cur_hi = y1, y2
            elif y1 > cur_hi:
                covered += cur_hi - cur_lo
                cur_lo, cur_hi = y1, y2
            else:
                cur_hi = max(cur_hi, y2)
        if cur_hi is not None:
            covered += cur_hi - cur_lo

        area = (area + (xb - xa) * covered) % MOD
    return area
```

```cpp
int rectangleArea(std::vector<std::vector<int>>& rectangles) {
    const long long MOD = 1000000007LL;
    std::vector<long long> xs;
    for (const auto& r : rectangles) {
        xs.push_back(r[0]);
        xs.push_back(r[2]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    long long area = 0;
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        long long xa = xs[i], xb = xs[i + 1];
        std::vector<std::pair<long long, long long>> spans;
        for (const auto& r : rectangles) {
            if (r[0] <= xa && xb <= r[2]) {
                spans.push_back({r[1], r[3]});
            }
        }
        std::sort(spans.begin(), spans.end());

        long long covered = 0;
        bool has = false;
        long long lo = 0, hi = 0;
        for (const auto& [y1, y2] : spans) {
            if (!has) {
                lo = y1;
                hi = y2;
                has = true;
            } else if (y1 > hi) {
                covered += hi - lo;
                lo = y1;
                hi = y2;
            } else {
                hi = std::max(hi, y2);
            }
        }
        if (has) {
            covered += hi - lo;
        }
        area = (area + (xb - xa) % MOD * (covered % MOD)) % MOD;
    }
    return static_cast<int>(area);
}
```

**为什么按竖带切分**：把所有矩形的左右边界作为切点，相邻两个切点之间是一条
「竖带」。竖带内部没有任何矩形的 x 边界，因此「哪些矩形覆盖这条带」是固定的。
于是对每条带，收集横向完全盖住它的矩形，把它们的 y 区间求并集得到覆盖高度，
带宽乘高度就是这条带的贡献。各条带互不重叠，直接累加。这是「扫描线 + 离散化」
最朴素的形态：不需要线段树，逐带重算即可。

- **复杂度**：x 去重后至多 `2n` 条带，每条带扫描 `n` 个矩形并排序，总体
  O(n^2 log n)；空间 O(n)。（本题 n ≤ 200，够用。）
- **易错点**：y 区间并集用「排序后合并」时，重叠与相邻都要并进去，判断写成
  `y1 > cur_hi` 才另起一段；结果要对 `1e9+7` 取模，C++ 里宽度与高度都先取模
  再相乘，避免溢出。
- **相似题**：391. 完美矩形（不重叠时的面积判定）；218. 天际线（同一套 x 扫描）。

### 391. 完美矩形（困难）

**题目**：判断一组矩形能否恰好（不重叠、无空隙）拼成一个大矩形。

**思路**：

```python
def is_rectangle_cover(rectangles):
    area = 0
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    corners = set()

    for x1, y1, x2, y2 in rectangles:
        area += (x2 - x1) * (y2 - y1)
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
        for pt in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
            if pt in corners:
                corners.remove(pt)
            else:
                corners.add(pt)

    if area != (max_x - min_x) * (max_y - min_y):
        return False
    expected = {(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)}
    return corners == expected
```

```cpp
bool isRectangleCover(std::vector<std::vector<int>>& rectangles) {
    long long area = 0;
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;
    std::set<std::pair<int, int>> corners;

    for (const auto& r : rectangles) {
        int x1 = r[0], y1 = r[1], x2 = r[2], y2 = r[3];
        area += 1LL * (x2 - x1) * (y2 - y1);
        minX = std::min(minX, x1);
        minY = std::min(minY, y1);
        maxX = std::max(maxX, x2);
        maxY = std::max(maxY, y2);
        std::pair<int, int> pts[4] = {{x1, y1}, {x1, y2}, {x2, y1}, {x2, y2}};
        for (const auto& p : pts) {
            if (corners.count(p)) {
                corners.erase(p);
            } else {
                corners.insert(p);
            }
        }
    }

    if (area != 1LL * (maxX - minX) * (maxY - minY)) {
        return false;
    }
    std::set<std::pair<int, int>> expected{
        {minX, minY}, {minX, maxY}, {maxX, minY}, {maxX, maxY}};
    return corners == expected;
}
```

**为什么「面积 + 角点奇偶」就够**：恰好铺满需要两个不变量同时成立。
第一，**总面积等于外接大矩形的面积**（左下角取全局最小、右上角取全局最大），
它排除了「有空隙」。
第二，把每个矩形的四个角点拿来不断「异或」（出现两次就抵消）。
在完美铺法里，内部拼接点都是偶数个矩形的公共角、会抵消光，只有大矩形的四个
外角各出现一次而留下，所以最终点集恰好是那四个角。它排除了「有重叠」：重叠处
会多出奇点或改变外角。两个条件一起，就是充要条件。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：角点要用「集合的对称差」而不是简单累加；面积在坐标大时要用 64 位
  （C++ 乘 `1LL`）；`expected` 是四个角的集合，必须完全相等，多一个、少一个
  都不行。
- **相似题**：850. 矩形面积 II（求并集面积）；223. 矩形面积（两个矩形的容斥，
  见第 28 篇）。

---

## 规律总结

1. **区间题先排序，再扫描**。228 / 57 / 1288 / 1851 / 218 / 850 全都是「排序 +
   一趟线性扫描」。排序键决定了扫描时能利用什么单调性：合并按左端升序、去覆盖
   按「左升右降」、事件按 x 升序。

2. **「当前维护的区间」是合并类题型的心跳**。57 的 `[start, end]`、1288 的
   `max_end`、352 的邻居合并，本质都是维护一个边界并向右推进。判断「接上」还是
   「另起」时，一个等号之差就会出错，要写清楚是闭还是开。

3. **双指针求交/差时，只让右端小的那一个前进**。986 与 1272 共享这个动作：
   右端小的区间已经用尽，不可能再与后面相交。这是区间版的双指针，与第 01 篇
   「数组与双指针」的对撞思路同源。

4. **动态集合题：二分定位 + 只动邻居**。352 说明新元素最多影响左右各一个区间，
   定位后用 O(1) 次局部修改完成；这与第 19 篇「设计题」里「用组合好的基础结构
   把每个操作压到均摊 O(1)」是同一种设计哲学。

5. **扫描线 = 事件排序 + 维护「当前有效集合」**。218 用大顶堆维护当前楼高、
   1851 用最小堆维护当前区间长度，都是「按坐标推进、堆里放活跃对象、用右端点做
   惰性删除」。堆里存什么（高度 / 长度）由「要问什么最优」决定。

6. **面积类问题靠离散化把连续变离散**。850 用所有 x 边界切竖带，带内覆盖关系
   不变，于是每条带化成一次「区间并集」；391 更进一步，用「总面积 + 角点奇偶」
   两个整数不变量，连扫描都不用。能用不变量判定的，就别真的去构造图形。

7. **区间端点的开闭是本篇最大的坑**。求交 `lo <= hi`、插入重叠
   `intervals[i][0] <= end`、删除不相交 `b <= lo`、扫描线中「离开」用
   `heap.top().right <= x`——这些等号的位置各不相同，但都能回到一句话：
   「闭区间里，端点算数」。写代码前先把每个不等号想清楚。

8. **与其它篇的联系**：区间合并的排序思想与第 14 篇「贪心」的按右端点排序
   （435 / 452）血脉相同；排 x 坐标切竖带与第 04 篇「前缀和与差分」里
   「差分数组按坐标累加」是同一套离散化；1851 / 218 的堆与第 08 篇「堆和 Top-K」
   的多路维护一脉相承；391 的面积容斥则可追溯到第 28 篇「计算几何」的开篇。
