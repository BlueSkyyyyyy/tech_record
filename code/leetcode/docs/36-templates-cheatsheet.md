# 模板速查与易错清单（附录）

前面三十五篇按题型展开，这一页反过来：**只留骨架，供动手前扫一眼、写完题对一遍。**
它不替代讲解——每个模板后面都标了出处，卡住时回到对应那一篇。

## 怎么用这一页

1. **先判类型**：看数据范围（下一节）→ 看题目问的是「可行性 / 最值 / 方案数」→ 选模板。
2. **只背骨架和边界**：模板的价值在「循环怎么写、哪个指针什么时候动」，不在变量名。
3. **写完对清单**：最后一节列了跨题型的常见错误，交卷前逐条过一遍。

## 一、按数据范围选复杂度

先看 n 的上限，基本能反推出该用什么算法。这是最省时间的「题型识别器」。

| n 的范围 | 可接受的复杂度 | 常见做法 |
|---|---|---|
| n ≤ 10 | O(n!)、O(2ⁿ · n) | 全排列、暴力枚举 |
| n ≤ 20 | O(2ⁿ)、O(2ⁿ · n) | 状压 DP、子集枚举 |
| n ≤ 100 | O(n³) | 区间 DP、Floyd |
| n ≤ 500 | O(n³) 偏紧、O(n²) 稳 | 区间 DP + 前缀和 |
| n ≤ 5×10³ | O(n²) | 线性 DP、双层枚举 |
| n ≤ 10⁵ | O(n log n) | 排序、二分、堆、树状数组、线段树 |
| n ≤ 10⁶ | O(n)、O(n log n) 偏紧 | 双指针、哈希、单调栈 / 队列 |
| n ≤ 10⁷ | O(n) | 线性扫描、位运算 |

> 反过来也成立：如果推出 O(n²) 而 n 是 10⁵，说明还缺一个「把内层循环换成数据结构」的优化
> （见「动态规划（二）」篇）。

## 二、模板卡

### 1. 二分查找：两个边界

**适用信号**：有序数组里找「第一个 ≥ x」「最后一个 ≤ x」的位置。

```python
def lower_bound(a, x):          # 第一个 >= x 的下标，可能等于 len(a)
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):          # 第一个 > x 的下标
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

**边界**：区间统一用「左闭右开」就不会差一；`while lo < hi` 配合 `hi = mid` 不会死循环。
**出处**：「二分查找」篇。

### 2. 答案二分

**适用信号**：题目问「最小的最大值 / 最大的最小值」，且给定一个答案能 O(n) 判断可行性。

```python
def solve(lo, hi, feasible):
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid        # mid 可行，答案在左半边（含 mid）
        else:
            lo = mid + 1
    return lo
```

**边界**：`feasible` 必须关于答案单调（通常是「越大越容易满足」）；上下界要取到真实答案的范围。
**出处**：「二分查找」篇（875、1011）。

### 3. 变长滑动窗口

**适用信号**：连续子数组 / 子串，求「最长 / 最短 / 个数」，窗口的合法性随右端扩张而变差。

```python
def longest_window(a):
    left = 0
    best = 0
    for right, x in enumerate(a):
        # 把 x 纳入窗口
        while not valid():          # 不合法就收缩左端
            # 把 a[left] 移出窗口
            left += 1
        best = max(best, right - left + 1)
    return best
```

**边界**：先扩张再收缩；「以 right 结尾」的计数方式与「最长」不同，别混用。
**出处**：「滑动窗口」篇。

### 4. 单调栈：下一个更大元素

**适用信号**：找每个元素左边 / 右边第一个比它大（小）的元素，或「以它为最低点的区间」。

```python
def next_greater(a):
    n = len(a)
    res = [-1] * n
    st = []                      # 存下标，栈内对应值递减
    for i, x in enumerate(a):
        while st and a[st[-1]] < x:
            res[st.pop()] = i
        st.append(i)
    return res
```

**边界**：栈里存下标而不是值（要算距离）；循环数组把数组接一遍（`i % n`）或走两轮。
**出处**：「栈与单调栈」篇。

### 5. 单调队列：窗口最大值

**适用信号**：固定大小窗口内反复取最值，且窗口在滑动。

```python
from collections import deque


def max_sliding_window(a, k):
    dq = deque()                 # 存下标，对应值递减
    out = []
    for i, x in enumerate(a):
        while dq and a[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:       # 队首过期
            dq.popleft()
        if i >= k - 1:
            out.append(a[dq[0]])
    return out
```

**边界**：先踢队尾再入队，最后才判队首过期；`<=` 与 `<` 决定重复值保留哪一个。
**出处**：「队列与双端队列」篇（239）、「动态规划（二）」篇（1425、1696）。

### 6. 并查集

**适用信号**：「连通 / 同组 / 等价」关系会传递，需要合并与查询。

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False          # 已经在同一集合
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
```

**边界**：`union` 返回 `False` 正好用来判环（冗余连接）；先合并再统计，别边合并边数。
**出处**：「并查集」篇。

### 7. 堆：Top-K

**适用信号**：求前 K 大 / 前 K 小 / 第 K 个，数据流或数据量大到不能全排序。

```python
import heapq


def top_k_largest(a, k):
    heap = []                    # 小顶堆，堆顶是当前第 K 大
    for x in a:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap
```

**边界**：求「最大」用最小堆，求「最小」用最大堆（存相反数）；`k` 大于长度时返回全部。
**出处**：「堆与 Top-K」篇。

### 8. 拓扑排序

**适用信号**：有依赖关系的任务排序、判「能不能排完」（有环则不能）。

```python
from collections import deque


def topo_sort(n, edges):
    graph = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:           # u -> v
        graph[u].append(v)
        indeg[v] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []
```

**边界**：返回空表示有环；要求字典序最小时把队列换成堆。
**出处**：「图与搜索」篇。

### 9. Dijkstra

**适用信号**：非负边权单源最短路。

```python
import heapq


def dijkstra(n, graph, src):
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue             # 过期条目，跳过
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
```

**边界**：负权不能用；`d > dist[u]` 的跳过不能省；「方案数 / 最短路计数」要在松弛时一起维护。
**出处**：「最短路径」篇。

### 10. 回溯：子集与排列

**适用信号**：枚举所有组合 / 排列 / 满足约束的构造。

```python
def subsets(nums):
    res = []
    path = []

    def dfs(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

    dfs(0)
    return res


def permutations(nums):
    res = []
    path = []
    used = [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return res
```

**边界**：组合用 `start` 只往后选，排列用 `used`；有重复元素时先排序，再用
`i > start and nums[i] == nums[i-1]`（组合）或 `i > 0 and nums[i] == nums[i-1] and not used[i-1]`（排列）去重。
**出处**：「回溯与递归」篇。

### 11. 背包：0/1 与完全

**适用信号**：从一组物品里选，满足容量 / 目标和约束，求最值或方案数。

```python
def knapsack_01(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):   # 倒序：每件只用一次
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


def knapsack_complete(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(w, capacity + 1):       # 正序：可重复取
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]
```

**边界**：0/1 必须倒序，完全背包正序；求「方案数」时把 `max` 换成 `+`，且 `dp[0] = 1`；
组合数与排列数的循环顺序不同（物品在外 vs 容量在外）。
**出处**：「动态规划（一）」篇。

### 12. 树状数组

**适用信号**：单点修改 + 前缀查询；或值域计数（数逆序对、数「比我小的有几个」）。

```python
class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s
```

**边界**：下标从 1 开始；值域大或含负数先离散化；区间加、单点查要把它架在差分数组上。
**出处**：「树状数组」篇。

### 13. 线段树：单点改 + 区间查

**适用信号**：区间查询的信息可合并（和、最值、gcd、最长连续段），且带修改。

```python
class SegTree:
    def __init__(self, a):
        self.n = len(a)
        self.tree = [0] * (4 * self.n)
        self._build(a, 1, 0, self.n - 1)

    def _build(self, a, node, lo, hi):
        if lo == hi:
            self.tree[node] = a[lo]
            return
        mid = (lo + hi) // 2
        self._build(a, node * 2, lo, mid)
        self._build(a, node * 2 + 1, mid + 1, hi)
        self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1]

    def update(self, idx, val, node=1, lo=0, hi=None):
        if hi is None:
            hi = self.n - 1
        if lo == hi:
            self.tree[node] = val
            return
        mid = (lo + hi) // 2
        if idx <= mid:
            self.update(idx, val, node * 2, lo, mid)
        else:
            self.update(idx, val, node * 2 + 1, mid + 1, hi)
        self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1]

    def query(self, ql, qr, node=1, lo=0, hi=None):
        if hi is None:
            hi = self.n - 1
        if qr < lo or hi < ql:
            return 0
        if ql <= lo and hi <= qr:
            return self.tree[node]
        mid = (lo + hi) // 2
        return self.query(ql, qr, node * 2, lo, mid) + self.query(
            ql, qr, node * 2 + 1, mid + 1, hi
        )
```

**边界**：数组开 `4n`；区间修改要加懒标记；「在哪 merge」决定 `pushup` 写在哪；
`query` 越界返回的是**单位元**——求和用 0、求最小值用 `+∞`，换成别的合并运算时要跟着改。
**出处**：「线段树」篇。

### 14. KMP：前缀函数

**适用信号**：单模式串匹配、判重复子串、求最长公共前后缀。

```python
def prefix_function(p):
    pi = [0] * len(p)
    for i in range(1, len(p)):
        j = pi[i - 1]
        while j > 0 and p[i] != p[j]:
            j = pi[j - 1]
        if p[i] == p[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(text, pattern):
    if not pattern:
        return 0
    pi = prefix_function(pattern)
    j = 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pattern[j]:
            j = pi[j - 1]
        if ch == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - len(pattern) + 1
    return -1
```

**边界**：回退是 `j = pi[j-1]`，不是 `j -= 1`；`pi[i]` 是「以 i 结尾的最长公共前后缀长度」。
**出处**：「字符串匹配」篇。

### 15. 树形 DP：两状态

**适用信号**：父子之间有「选 / 不选」的互斥约束，要在整棵树上求最优。

```python
def tree_dp(root):
    def dfs(node):
        if node is None:
            return (0, 0)        # (选, 不选)
        l_take, l_skip = dfs(node.left)
        r_take, r_skip = dfs(node.right)
        take = node.val + l_skip + r_skip
        skip = max(l_take, l_skip) + max(r_take, r_skip)
        return (take, skip)

    return max(dfs(root))
```

**边界**：返回值含义要在函数第一行写明；全局最优若在中间节点结算，记得每个节点都更新一次。
**出处**：「树形 DP」篇。

### 16. 区间 DP：枚举最后一步

**适用信号**：答案能写成「把区间切成两段」或「最后处理某个元素」，如戳气球、合并石头、回文分割。

```python
def interval_dp(a):
    n = len(a)
    pre = [0] * (n + 1)                      # 前缀和，O(1) 取区间代价
    for i, x in enumerate(a):
        pre[i + 1] = pre[i] + x

    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):           # 区间长度从小到大
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = min(
                dp[i][k] + dp[k + 1][j] for k in range(i, j)
            ) + (pre[j + 1] - pre[i])        # 合并成本 = 区间和
    return dp[0][n - 1]
```

**边界**：按长度而不是按下标遍历，保证子区间先算完；`cost` 常配前缀和；「最后一步」的定义决定
`k` 的含义，先想清楚再写。

## 三、跨题型易错清单

1. **空输入 / 单元素 / k 大于长度**：每道题先想这三个边界，再写主逻辑。
2. **二分死循环**：`while lo < hi` 配 `hi = mid`，或 `while lo <= hi` 配 `lo = mid + 1`；两套不能混。
3. **整数溢出**：`mid = (lo + hi) // 2` 在 C++ 里可能溢出，写 `lo + (hi - lo) / 2`；乘积、前缀和、
   答案累加按需用 `long long`。
4. **`INT_MIN` 取绝对值**：`abs(INT_MIN)` 是未定义行为，先转 `long long`。
5. **Python 的引用**：`res.append(path)` 存的是同一个列表，要 `path[:]`；二维数组别写
   `[[0] * n] * m`（行是同一个对象）。
6. **C++ 的迭代器失效**：`vector` 扩容后旧迭代器 / 引用失效；边遍历边删除用 `it = erase(it)`。
7. **递归深度**：Python 默认 1000，深链会 `RecursionError`，可改写成迭代或 `sys.setrecursionlimit`。
8. **去重的两处**：数组去重要「同层跳过」，组合去重要先排序；漏一处就重复或超时。
9. **字符串切片的隐藏 O(n)**：循环里反复 `s[:i]`、`s += ch`、`list.insert(0, x)`、`x in list`
   都会把 O(n) 变成 O(n²)。
10. **浮点比较**：不要用 `==`，用 `abs(a - b) < 1e-9`；能转整数（乘 10^k）就转。
11. **取模的负数**：Python 的 `%` 结果非负，C++ 的 `%` 可能为负，需要 `(x % m + m) % m`。
12. **哈希的最坏情况**：`unordered_map` 可能被卡到 O(n)，竞赛里可换 `map` 或加随机盐；Python dict
    一般安全，但别依赖遍历顺序。
13. **图的重边与自环**：建图前想清楚要不要去重；自环会让入度、判环、最短路都出问题。
14. **并查集的递归 `find`**：链很长时递归会爆栈，模板用迭代 + 路径压缩更稳。
15. **随机题的种子**：`random` 默认按时间播种，自测要固定种子才能复现；不要在循环里重复 `seed()`。

## 四、C++ 对照要点

- `priority_queue<int>` 是**大顶堆**；小顶堆写 `priority_queue<int, vector<int>, greater<int>>`。
- `lower_bound / upper_bound` 返回迭代器，减 `begin()` 才是下标；对 `vector` 要求已排序。
- `unordered_map` 的 `operator[]` 会插入默认值，只查不改用 `find` / `count`。
- 结构化绑定（`auto [a, b] = ...`）需要 C++17；本仓库统一 `-std=c++17`。
- 自定义比较器用 lambda 时注意「严格弱序」：相等必须返回 `false`。
- `assert(v == std::vector<int>{1, 2})` 会被宏的逗号拆坏，先把期望值存进变量。
- `char` 在多数平台有符号，`isalnum / tolower` 前先转 `unsigned char`。

## 五、回到各篇

模板只是索引，真正的判断力来自「为什么这样移动指针 / 为什么这个状态够用」。按下面的顺序回看：

- 数组、字符串、链表、栈、队列 → 基础操作与双指针；
- 二分、排序、堆、哈希 → 「把 O(n) 查找换成 O(log n) / O(1)」；
- 二叉树、图、回溯、分治 → 递归与搜索的四种骨架；
- 动态规划（一）（二）、树形 DP、数位 DP、状压 → 状态设计与优化；
- 并查集、树状数组、线段树、最短路径 → 用结构维护关系与距离。
