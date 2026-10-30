# 动态规划（二）：给 DP 提速降维

动态规划（一）解决了「状态怎么定义、转移怎么写」。但很多时候，状态和转移都对，
复杂度却不达标——比如 `dp[i] = 值 + max(dp[j])` 朴素转移是 O(n²)，数据一大就超时。
本篇专门讲**优化 DP** 的几件常用武器：把转移里的「求和 / 求最大值 / 找前驱」从 O(n) 降到 O(1) 或 O(log n)，
以及用**滚动数组**把空间从 O(n²) 压到 O(n)。

优化 DP 有个前提：**先把朴素的 DP 写对，再看转移里到底哪一步慢**。
本篇每一题都遵循这个顺序——先给朴素式，再指出瓶颈，最后换武器。

本篇题目（由易到难）：

| 优化手段 | 题目 | 难度 |
|---|---|---|
| 前缀和优化区间求和 | 813. 最大平均值和的分组 | 中等 |
| 哈希表把「找前驱」降到 O(1) | 1218. 最长定差子序列 | 中等 |
| 二分查找定位「可接的前一个」 | 1235. 规划兼职工作 | 困难 |
| 二分 + 多一维「场次上限」 | 1751. 最多可以参加的会议数目 II | 困难 |
| 二分找「交点」优化 min-max 转移 | 887. 鸡蛋掉落 | 困难 |
| 单调队列优化「窗口最大值」 | 1425. 带限制的子序列和 | 困难 |
| 单调队列优化「窗口最大值」 | 1696. 跳跃游戏 VI | 中等 |
| 滚动数组做空间压缩 | 174. 地下城游戏 | 困难 |

> 一句话记住：**DP 优化不是换状态，而是换「取 max / 求和 / 找前驱」这些转移动作的实现方式。**
> 窗口固定、只求最值 → 单调队列；求和 → 前缀和；找特定值 → 哈希表；找区间位置 → 二分。

---

## 模式一：前缀和优化「区间求和」

**适用信号**：转移里出现「一段区间的和 / 平均值」，且这段区间的边界随状态滑动。

**核心动作**：先用前缀和 `pre[i] = a[0] + … + a[i-1]` 把任意区间和变成 `pre[r] - pre[l]`，
把原本 O(n) 的区间求和降成 O(1)。注意它优化的是「转移里的求和」，状态本身不变。

### 813. 最大平均值和的分组（中等）

**题目**：把数组 `nums` 分成最多 `k` 段非空连续子数组，使各段「平均值之和」最大。

**思路**：设 `dp[j][i]` = 前 `i` 个数分成 `j` 段的最大平均值和。枚举最后一段的起点 `m`：

$$
dp[j][i] = \max_{m}\left(dp[j-1][m] + \frac{pre[i]-pre[m]}{i-m}\right)
$$

若没有前缀和，每次算 `(pre[i]-pre[m])/(i-m)` 都要重新累加区间和；有了 `pre`，
区间和是 O(1)，瓶颈就只剩「枚举 m」这一层。边界 `dp[1][i] = pre[i] / i`。

**代码**（`src/dp-optimization/largest_sum_of_averages.py` / `.cpp`）：

```python
def largest_sum_of_averages(nums, k):
    n = len(nums)
    pre = [0] * (n + 1)
    for i, v in enumerate(nums):
        pre[i + 1] = pre[i] + v

    dp = [[0.0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        dp[1][i] = pre[i] / i
    for j in range(2, k + 1):
        for i in range(j, n + 1):
            best = 0.0
            for m in range(j - 1, i):
                cand = dp[j - 1][m] + (pre[i] - pre[m]) / (i - m)
                if cand > best:
                    best = cand
            dp[j][i] = best
    return dp[k][n]
```

```cpp
double largestSumOfAverages(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<double> pre(n + 1, 0.0);
    for (int i = 0; i < n; ++i) pre[i + 1] = pre[i] + nums[i];

    std::vector<std::vector<double>> dp(k + 1, std::vector<double>(n + 1, 0.0));
    for (int i = 1; i <= n; ++i) dp[1][i] = pre[i] / i;
    for (int j = 2; j <= k; ++j) {
        for (int i = j; i <= n; ++i) {
            double best = 0.0;
            for (int m = j - 1; m < i; ++m) {
                double cand = dp[j - 1][m] + (pre[i] - pre[m]) / (i - m);
                if (cand > best) best = cand;
            }
            dp[j][i] = best;
        }
    }
    return dp[k][n];
}
```

- **复杂度**：时间 O(k·n²)，空间 O(k·n)。前缀和只是把最内层的加法常数降下来，主循环结构未变。
- **易错点**：`dp` 用 1-indexed 表示「前 i 个」，`pre` 下标要和它对齐；平均值用浮点除法。
- **相似题**：1478. 安排邮筒、410. 分割数组的最大值（都是「划分 + 区间代价」，常配前缀和）、
  304. 二维区域和检索（二维前缀和，见前缀和篇）。

---

## 模式二：哈希表把「找前驱」降到 O(1)

**适用信号**：转移要「往前找一个值为 X 的元素」，而这个「X」完全由当前值算出，与下标无关。

**核心动作**：把 DP 状态按**值**存进哈希表，而不是按**下标**数组。扫描到 `x` 时，
直接 `查表(x 的目标前驱)`，O(1) 拿到，替代 O(n) 的回溯。

### 1218. 最长定差子序列（中等）

**题目**：给定数组 `arr` 和整数 `difference`，求最长的等差子序列（相邻两项差恰为 `difference`）长度。

**思路**：朴素式 `dp[i] = 1 + max{dp[j] : arr[j] = arr[i] - difference, j < i}`，往前扫是 O(n²)。
但「前驱」只由**值**决定：用一个字典记「以某值为结尾的最长长度」，扫描时直接查
`dp.get(x - difference, 0) + 1`。重复值会自然合并（保留最长），一次遍历即可。

**代码**（`src/dp-optimization/longest_arithmetic_subsequence.py` / `.cpp`）：

```python
def longest_subsequence(arr, difference):
    dp = {}
    best = 0
    for x in arr:
        dp[x] = dp.get(x - difference, 0) + 1
        best = max(best, dp[x])
    return best
```

```cpp
int longestSubsequence(const std::vector<int> &arr, int difference) {
    std::unordered_map<int, int> dp;
    int best = 0;
    for (int x : arr) {
        auto it = dp.find(x - difference);
        int len = (it == dp.end() ? 0 : it->second) + 1;
        dp[x] = len;
        best = std::max(best, len);
    }
    return best;
}
```

- **复杂度**：时间 O(n)（哈希均摊），空间 O(n)。
- **易错点**：键是**值**不是下标；要 `max(best, dp[x])`，因为最长子序列不一定以最后一个元素结尾；`difference` 可为负或 0。
- **相似题**：300. 最长递增子序列（把「找前驱」换成「找区间最大值」，那要用线段树 / 树状数组优化）、
  560. 和为 K 的子数组（哈希存前缀和，见前缀和篇）、1027. 最长等差数列（哈希表按「差值」分层）。

---

## 模式三：二分查找定位「可接的前一个」

**适用信号**：要选一批**不重叠区间**使收益最大，转移是「接在某个结束时间不冲突的区间之后」。

**核心动作**：按结束时间排序后，用二分查找 O(log n) 找到「最后一个结束时间 ≤ 当前开始时间」
的区间，替代 O(n) 的线性回溯。

### 1235. 规划兼职工作（困难）

**题目**：给出 `startTime / endTime / profit`，选不重叠的若干工作使总报酬最大。

**思路**：按结束时间排序，`dp[i]` = 只考虑前 `i` 个工作的最大报酬：
- 不做第 `i` 个：`dp[i-1]`；
- 做第 `i` 个：找到最后一个「结束时间 ≤ 它的开始时间」的工作 `m`，则 `dp[m] + p`。
排序后结束时间有序，用 `bisect_right(ends, s, 0, i-1)` 二分定位 `m`。

**代码**（`src/dp-optimization/job_scheduling.py` / `.cpp`）：

```python
import bisect

def job_scheduling(start_time, end_time, profit):
    jobs = sorted(zip(end_time, start_time, profit))
    ends = [e for e, _, _ in jobs]
    n = len(jobs)
    dp = [0] * (n + 1)
    for i, (_e, s, p) in enumerate(jobs, 1):
        m = bisect.bisect_right(ends, s, 0, i - 1)
        dp[i] = max(dp[i - 1], dp[m] + p)
    return dp[n]
```

```cpp
int jobScheduling(const std::vector<int> &startTime, const std::vector<int> &endTime,
                  const std::vector<int> &profit) {
    int n = static_cast<int>(startTime.size());
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return endTime[a] < endTime[b]; });

    std::vector<int> ends(n);
    for (int i = 0; i < n; ++i) ends[i] = endTime[idx[i]];

    std::vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        int j = idx[i - 1];
        int s = startTime[j], p = profit[j];
        // ends[0..i-2] 中 <= s 的个数 = 可接的最后一个工作下标 + 1
        int m = static_cast<int>(std::upper_bound(ends.begin(), ends.begin() + (i - 1), s) -
                                 ends.begin());
        dp[i] = std::max(dp[i - 1], dp[m] + p);
    }
    return dp[n];
}
```

- **复杂度**：时间 O(n log n)（排序 + 每个工作一次二分），空间 O(n)。
- **易错点**：先排序再二分；二分范围是「前 i-1 个」；开始时刻等于结束时刻不算重叠，用 `bisect_right` / `upper_bound`。
- **相似题**：435. 无重叠区间、646. 最长数对链（贪心解，见贪心篇）；1751. 最多可以参加的会议数目 II（本题加一维「场次上限」）。

### 1751. 最多可以参加的会议数目 II（困难）

**题目**：`events[i] = [start, end, value]`，同一时间只能参加一个会议（后一个的 `start` 必须严格
大于前一个的 `end`）。最多参加 `k` 个会议，求能获得的最大价值。

**思路（在 1235 上加一维「选了几个」）**：
按结束时间排序后，`dp[i][j]` = 只考虑前 `i` 个会议、最多选 `j` 个时的最大价值。对第 `i` 个会议：

- 不选：`dp[i-1][j]`；
- 选：`dp[p][j-1] + value_i`，其中 `p` 是「结束时间 < start_i」的最后一个会议下标。

`p` 用二分求（结束时间已排序），把朴素 O(n) 的「枚举前驱」压成 O(log n)。状态从两维变成三维的
「前 i 个 × 选 j 个」，但转移里那次查找还是同一个二分——**加维度不一定要换优化手段**。

**代码**（`src/dp-optimization/maximum_number_of_events_that_can_be_attended_ii.py` / `.cpp`）：

```python
from bisect import bisect_left


def max_value(events, k):
    events = sorted(events, key=lambda e: e[1])
    n = len(events)
    ends = [e[1] for e in events]

    # dp[i][j]：前 i 个会议、最多选 j 个
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        start, _end, value = events[i - 1]
        p = bisect_left(ends, start)  # 结束时间 < start 的会议个数
        for j in range(1, k + 1):
            dp[i][j] = max(dp[i - 1][j], dp[p][j - 1] + value)
    return dp[n][k]
```

```cpp
int maxValue(std::vector<std::vector<int>> events, int k) {
    std::sort(events.begin(), events.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) { return a[1] < b[1]; });
    int n = static_cast<int>(events.size());
    std::vector<int> ends(n);
    for (int i = 0; i < n; ++i) ends[i] = events[i][1];

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(k + 1, 0));
    for (int i = 1; i <= n; ++i) {
        int start = events[i - 1][0];
        int value = events[i - 1][2];
        int p = static_cast<int>(std::lower_bound(ends.begin(), ends.end(), start) - ends.begin());
        for (int j = 1; j <= k; ++j) {
            dp[i][j] = std::max(dp[i - 1][j], dp[p][j - 1] + value);
        }
    }
    return dp[n][k];
}
```

- **复杂度**：排序 O(n log n)，DP O(n · k · log n)，空间 O(n · k)。
- **易错点**：这里用的是 `bisect_left` / `lower_bound`（严格小于 `start`），与 1235 的
  `bisect_right`（小于等于）不同——因为本题要求后一个会议的开始**严格大于**前一个的结束；
  `dp[p][j-1]` 的 `p` 是「会议个数」，恰好也是「前 p 个」的下标，不用再 `+1`。
- **相似题**：1235（k = 1 的退化版）；300 的「前驱」查找；410. 分割数组的最大值（另一类
  「加一维 + 二分」的组合）。

---

## 模式四：单调队列优化「窗口最大值」

**适用信号**：转移形如 `dp[i] = a[i] + max{dp[j] : i-k ≤ j < i}`，即**固定大小窗口求最值**。

**核心动作**：用**单调递减双端队列**维护窗口内的候选下标，队首恒为当前窗口最大值。
每个元素最多进出队一次，整体 O(n)。

> 为什么不是堆？堆取最值 O(log n)，且「过期元素」只能在堆顶检查，删除麻烦；
> 单调队列对**滑动窗口最值**是 O(n) 最优解。

### 1425. 带限制的子序列和（困难）

**题目**：选非空子序列，相邻元素下标差 ≤ k，求最大子序列和。

**思路**：`dp[i] = nums[i] + max(0, max{dp[j] : i-k ≤ j < i})`，`max(0,…)` 表示可以从 `nums[i]` 重新开始。
窗口 `[i-k, i-1]` 固定，用单调递减队列维护 `dp` 值。

**代码**（`src/dp-optimization/constrained_subsequence_sum.py` / `.cpp`）：

```python
from collections import deque

def constrained_subset_sum(nums, k):
    n = len(nums)
    dp = [0] * n
    dq = deque()
    best = nums[0]
    for i in range(n):
        while dq and dq[0] < i - k:
            dq.popleft()
        prev = max(0, dp[dq[0]]) if dq else 0
        dp[i] = nums[i] + prev
        best = max(best, dp[i])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return best
```

```cpp
int constrainedSubsetSum(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    std::deque<int> dq;
    int best = nums[0];
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) dq.pop_front();
        int prev = dq.empty() ? 0 : std::max(0, dp[dq.front()]);
        dp[i] = nums[i] + prev;
        best = std::max(best, dp[i]);
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：先「踢过期队首」再取值；`best` 不能只取 `dp[n-1]`（子序列不一定以末尾结束）；`max(0,…)` 别忘。
- **相似题**：1696（下一题）；239. 滑动窗口最大值（单调队列的原型，见队列篇）；862. 和至少为 K 的最短子数组。

### 1696. 跳跃游戏 VI（中等）

**题目**：从下标 0 出发，每次最多向右跳 `k` 步，最后到达 `n-1`，得分是经过位置的 `nums` 之和，求最大得分。

**思路**：`dp[i] = nums[i] + max{dp[j] : i-k ≤ j < i}`，`dp[0] = nums[0]`。与上一题几乎同构，
区别是没有 `max(0,…)`（必须从起点接过来），且答案是 `dp[n-1]`。

**代码**（`src/dp-optimization/jump_game_vi.py` / `.cpp`）：

```python
from collections import deque

def max_result(nums, k):
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])
    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()
        dp[i] = nums[i] + dp[dq[0]]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return dp[n - 1]
```

```cpp
int maxResult(const std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    dp[0] = nums[0];
    std::deque<int> dq;
    dq.push_back(0);
    for (int i = 1; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) dq.pop_front();
        dp[i] = nums[i] + dp[dq.front()];
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return dp[n - 1];
}
```

- **复杂度**：时间 O(n)，空间 O(n)（可进一步压到 O(k)）。
- **易错点**：窗口是 `[i-k, i-1]`，所以「踢过期队首」的条件是 `dq[0] < i - k`（等于 `i-k` 仍可用）；
  这里的 `dp[i]` 必须是「落到 i 的最大得分」，不能用 `max(0, …)` 截断——与 1425 允许重新开始不同，
  本题每一格都必须踩；`dp[0]` 要单独初始化。
- **相似题**：1425（上一题）、45. 跳跃游戏 II（贪心 / BFS 另一种思路，见贪心篇）、1186. 删除一次得到子数组最大和（状态机 + 滑窗）。

---

## 模式五：滚动数组做空间压缩

**适用信号**：`dp[i][j]` 只依赖「上一行」或「相邻几个格子」，二维表其实不必全存。

**核心动作**：把二维 DP 压成一维。关键是**确定遍历方向**，保证用到的旧值在被覆盖前已经读过。
另外，「从右下角倒推」这类问题用滚动时要额外处理**边界哨兵**。

### 174. 地下城游戏（困难）

**题目**：骑士从左上走到右下（只能右 / 下），每格加血或掉血，任意时刻血量 ≥ 1，求最小初始血量。

**思路**：正向的「当前血量」会污染状态，改为**从终点倒推**：
`dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])`。
由于只依赖右边和下边，可用一维数组 `dp[j]`：更新前 `dp[j]` 是「下一行同列」，`dp[j+1]` 是「本行右一列」，
每行从右往左刷新；额外留一个 `dp[n]` 作为「右侧越界」哨兵。

**代码**（`src/dp-optimization/dungeon_game.py` / `.cpp`）：

```python
def calculate_minimum_hp(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[0] * n for _ in range(m)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if i == m - 1 and j == n - 1:
                need = 1 - dungeon[i][j]
            elif i == m - 1:
                need = dp[i][j + 1] - dungeon[i][j]
            elif j == n - 1:
                need = dp[i + 1][j] - dungeon[i][j]
            else:
                need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
            dp[i][j] = max(1, need)
    return dp[0][0]
```

```python
def calculate_minimum_hp_1d(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    INF = float("inf")
    dp = [INF] * (n + 1)
    dp[n - 1] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[j] = max(1, min(dp[j], dp[j + 1]) - dungeon[i][j])
        dp[n] = INF
    return dp[0]
```

- **复杂度**：二维 O(m·n) 空间；滚动版 O(n) 空间，时间同为 O(m·n)。
- **易错点**：必须**倒推**（正推的「到达该格时的血量」不能作为状态）；滚动版每行结束要把 `dp[n]` 重置为 `INF`（右侧越界没有格子）；`dp[n-1] = 1` 是终点的「右侧哨兵」。
- **相似题**：62 / 64 不同路径与最小路径和（正向网格 DP）、1143 最长公共子序列（二维压一维的经典）；
  只要依赖「左上角」就正序滚动、依赖「右下角」就倒序滚动。

---

## 模式六：二分找「交点」，优化 min-max 转移

**适用信号**：转移形如 `dp[i][j] = 1 + min_x max(f(x), g(x))`，其中 `f` 随 x 单调增、`g` 随 x 单调减。
这是「最坏情况最小化」的典型形状（鸡蛋掉落、分割数组、猜数字大小 II 的某些变体）。

**核心动作**：`max(递增, 递减)` 是**单峰**的，谷底就在两条线的交点附近。于是不用枚举 x，
而是二分「当前该往哪边走」：`f(mid) > g(mid)` 说明「选大了」，把右界收；否则左界抬。
二分过程中遇到的每个候选都拿来更新答案。

### 887. 鸡蛋掉落（困难）

**题目**：`k` 枚鸡蛋、`n` 层楼，存在临界楼层 `f`（≤ f 不碎，> f 碎）。每次选一层扔一枚蛋，
碎了就不能再用。求**保证**能确定 `f` 的最少实验次数。

**思路（DP + 二分优化转移）**：
设 `dp[i][j]` = 用 `i` 枚蛋、确定 `j` 层所需的最少次数。枚举第一次扔的楼层 `x`：

- 碎了 → 剩 `i-1` 枚蛋，答案在下面 `x-1` 层：`dp[i-1][x-1]`；
- 没碎 → 还有 `i` 枚蛋，答案在上面 `j-x` 层：`dp[i][j-x]`。

最坏情况取 max，再对所有 x 取 min：

```
dp[i][j] = 1 + min_x max(dp[i-1][x-1], dp[i][j-x])
```

直接枚举 x 是 O(k·n²)。注意第一项关于 x **递增**、第二项关于 x **递减**，`max` 是单峰的，
二分交点即可把内层降到 O(log n)。边界上 `dp[0][j] = ∞`（没有蛋就确定不了任何楼层），
否则「只有一枚蛋却从顶层扔」会被错误地算成 1 次。

**代码**（`src/dp-optimization/super_egg_drop.py` / `.cpp`）：

```python
def super_egg_drop(k, n):
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    for floors in range(1, n + 1):
        dp[0][floors] = float("inf")  # 没有蛋就确定不了任何楼层
    for eggs in range(1, k + 1):
        for floors in range(1, n + 1):
            lo, hi = 1, floors
            best = float("inf")
            while lo <= hi:
                mid = (lo + hi) // 2
                broken = dp[eggs - 1][mid - 1]
                intact = dp[eggs][floors - mid]
                best = min(best, 1 + max(broken, intact))
                if broken > intact:
                    hi = mid - 1
                else:
                    lo = mid + 1
            dp[eggs][floors] = best
    return dp[k][n]
```

```cpp
int superEggDrop(int k, int n) {
    // dp[eggs][floors]：eggs 枚蛋、确定 floors 层需要的最少次数
    const int INF = 1000000000;
    std::vector<std::vector<int>> dp(k + 1, std::vector<int>(n + 1, 0));
    for (int floors = 1; floors <= n; ++floors) dp[0][floors] = INF;

    for (int eggs = 1; eggs <= k; ++eggs) {
        for (int floors = 1; floors <= n; ++floors) {
            int lo = 1, hi = floors, best = INF;
            while (lo <= hi) {
                int mid = lo + (hi - lo) / 2;
                int broken = dp[eggs - 1][mid - 1];
                int intact = dp[eggs][floors - mid];
                int worst = 1 + std::max(broken, intact);
                if (worst < best) best = worst;
                if (broken > intact) {
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }
            dp[eggs][floors] = best;
        }
    }
    return dp[k][n];
}
```

- **复杂度**：时间 O(k · n log n)，空间 O(k · n)。
- **易错点**：`dp[0][j]` 必须是 `∞`，否则一枚蛋也能「一次搞定」；循环顺序是 `eggs` 外层、
  `floors` 内层，保证 `dp[eggs][floors-mid]` 已算（同一行、更小的 floors）；二分只用来定位谷底，
  答案要在二分过程中顺手取 min，不能只取最后一次。
- **相似题**：410. 分割数组的最大值、1011. 在 D 天内送达包裹（「答案二分 + 可行性判断」，见二分篇）；
  375. 猜数字大小 II（同样是 min-max 转移，但数据范围小，直接枚举即可）。

---

## 规律总结

1. **优化前先写朴素 DP**。看清转移里慢在哪一步：求和 → 前缀和；求固定窗口最值 → 单调队列；
   找特定值 → 哈希表；找「可接的前一个」 → 二分（配排序）。**换的是动作，不是状态。**
2. **窗口最值统一用单调队列**：队首出窗口就 `popleft`，入队前把队尾「不如自己」的弹掉，
   保持单调。它把 O(n·k) 降到 O(n)，是滑动窗口类 DP 的标配（1425、1696，原型是 239）。
3. **哈希表适合「按值状态」**：当 DP 只关心「某个值出现过的最优解」时，用字典按值存，
   天然合并重复、O(1) 查询（1218）。若关心的是「一段值域内的最值」，那要上树状数组 / 线段树。
4. **二分适合「有序 + 找位置」**：不重叠区间类问题先按结束时间排序，再二分找可接的区间（1235），
   把 O(n²) 的「枚举所有前驱」降成 O(n log n)；加一维「最多选 k 个」时，二分照旧（1751），
   **加维度不等于换优化手段**。
5. **滚动数组省的是空间，不是时间**：只要某一行只依赖上一行（或相邻格），
   就能把 O(n²) 压到 O(n)。**方向由依赖决定**：依赖左 / 上 → 正序，依赖右 / 下 → 倒序，
   并给越界补哨兵值。
6. **别过度优化**：如果朴素 DP 已经是 O(n) 或数据范围允许 O(n²)，就不要硬上数据结构——
   优化引入的常数和 bug 风险，常常不值得。先让它对，再让它快。
7. **min-max 转移看「单峰」**：`1 + min_x max(f(x), g(x))` 里若 f 增、g 减，就可以二分交点，
   把枚举 x 的 O(n) 降成 O(log n)（887）。这和「答案二分」不是一回事：那里二分的是答案，
   这里二分的是**转移里的决策变量**。
8. **优化到最后都是「换个数据结构存中间结果」**：前缀和存区间和、单调队列存窗口最值、
   哈希表存值 → 最优解、树状数组存值域计数。看到转移里反复求同一个量，先问「它能不能预存」。

> 延伸阅读：DP 的状态定义与基础转移见「动态规划（一）」；单调队列的原型见「队列与双端队列」；
> 前缀和的更多用法见「前缀和与差分」。
