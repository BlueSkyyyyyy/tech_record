# 树状数组：让「单点修改 + 前缀查询」都只需 log n

有些题目的数组是「活的」：一边改元素，一边问区间和。静态前缀和 O(1) 查询很爽，但改一个数就要
重算后面一整段；暴力改 O(1)、查询 O(n)。两种做法各牺牲一头。

**树状数组（Binary Indexed Tree, BIT，又叫 Fenwick 树）** 就是那个折中点：单点修改和前缀和查询
都能做到 O(log n)。它还有一个更常用的副业——**值域计数**：边扫描边统计「已经出现过的数里，
比 x 大 / 小的有多少个」，几乎所有「数逆序对 / 数符合条件的数对」的题都靠它。

## 模板先记牢

树状数组靠 `lowbit(i) = i & -i`（i 的二进制最低位的 1 所代表的值）把下标分层。原数组下标从 1
开始（0 号位空着），`tree[i]` 维护区间 `(i - lowbit(i), i]` 的和：

- **单点加**：从 i 出发不断 `i += lowbit(i)`，一路加上去；
- **前缀和**：从 i 出发不断 `i -= lowbit(i)`，一路累加上来。

两种走法最多各走 O(log n) 步。下面是贯穿本篇的模板。

Python：

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

C++：

```cpp
struct Fenwick {
    int n;
    std::vector<long long> tree;
    explicit Fenwick(int n) : n(n), tree(n + 1, 0) {}
    void add(int i, long long delta) {
        for (; i <= n; i += i & -i) tree[i] += delta;
    }
    long long prefix(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += tree[i];
        return s;
    }
};
```

> 要理解它为什么对，盯住二进制就够：`prefix(i)` 其实是把 i 用 `i -= lowbit(i)` 拆成若干个
> 「分段和」，每段恰好是某个 `tree` 节点；`add` 则反过来，把改动沿着 `i += lowbit(i)` 传播给
> 所有包含它的节点。两者是逆过程，所以**不重不漏**。

## 本篇要解决的问题与模式

| 模式 | 树状数组扮演的角色 | 题目 | 难度 |
|---|---|---|---|
| 模式一：动态前缀和 | 单点修改 + 区间和 | 307. 区域和检索 - 数组可修改 | 中等 |
| 模式二：值域计数数逆序 | 边扫边数「比我大 / 小」 | LCR 170. 交易逆序对总数 / 1649. 通过指令创建有序数组 / 315. 计算右侧小于当前元素的个数 / 493. 翻转对 | 困难 |
| 模式三：前后各数一遍 | 左右计数相乘 | 1395. 统计作战单位数 / 2179. 统计数组中好三元组数目 | 中等/困难 |
| 模式四：前缀和 + 树状数组 | 把区间条件变成两个前缀之差 | 327. 区间和的个数 / 2426. 满足不等式的数对数目 | 困难 |
| 模式五：维护动态位置 | 用「占位」描述会移动的顺序 | 1409. 查询带键的排列 | 中等 |

模式二的四道题其实是**同一套动作**：离散化 + 从左（或从右）扫 + 查前缀和。把它们放在一起，
一眼就能看出「数逆序」「数右侧更小」「数 > 2x」只是换了查询的阈值方向。

---

## 模式一：动态前缀和

**适用信号**：数组会被**单点修改**，同时要频繁查询**区间和**。

**核心动作**：把「元素值」存进树状数组的值域坐标（这里是下标本身），修改时加差值，查询时做两次
前缀和相减。

### 307. 区域和检索 - 数组可修改（中等）

**题目**：实现一个数据结构，支持 `update(i, val)`（把 `nums[i]` 改成 val）和
`sumRange(l, r)`（返回闭区间 `[l, r]` 的元素和）。

**思路**：

前缀和数组能 O(1) 查询，但修改后要更新一整段后缀；普通数组修改 O(1)，查询却要 O(n)。树状数组
让两者都变 O(log n)：建树时把每个元素插进去；修改时只插入**差值** `val - nums[i]`（不是重新插
val），这样树状数组里存的仍是当前数组；查询 `[l, r]` 的和等于
`prefix(r + 1) - prefix(l)`，因为树状数组内部以下标 1 为起点。

**代码**：

```python
class NumArray:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.bit = Fenwick(self.n)
        for i, x in enumerate(nums):
            self.bit.add(i + 1, x)

    def update(self, index, val):
        self.bit.add(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sum_range(self, left, right):
        return self.bit.prefix(right + 1) - self.bit.prefix(left)
```

```cpp
class NumArray {
    std::vector<int> nums;
    Fenwick bit;

public:
    explicit NumArray(std::vector<int> nums_) : nums(std::move(nums_)), bit(nums.size()) {
        for (int i = 0; i < static_cast<int>(nums.size()); ++i) bit.add(i + 1, nums[i]);
    }

    void update(int index, int val) {
        bit.add(index + 1, static_cast<long long>(val) - nums[index]);
        nums[index] = val;
    }

    long long sumRange(int left, int right) {
        return bit.prefix(right + 1) - bit.prefix(left);
    }
};
```

- **复杂度**：建树 O(n log n)，`update` / `sumRange` 都是 O(log n)，空间 O(n)。
- **易错点**：树状数组下标从 1 开始，所有传参别忘了 `+1`；`update` 要传**差值**，直接传 val
  会把旧值也叠加上去；要保留一份 `nums` 才能算出差值。
- **相似题**：303 区域和检索 - 数组不可变（没有修改，前缀和即可，见 `docs/04-prefix-sum.md`）；
  304 二维区域和（二维前缀和，见 `docs/04-prefix-sum.md`）；本题是树状数组的「裸模板」。

---

## 模式二：值域计数数逆序

**适用信号**：要数「有多少个 (i, j) 满足 i < j 且两个数大小满足某种关系」。

**核心动作**：把**数值大小**当作树状数组的下标（值域坐标），从左往右（或从右往左）扫描，
每遇到一个数就先查「已出现过的数里满足条件的有几个」，再把当前数插进去。数值很大或为负时，
先**离散化**成 1..m 的排名。

### LCR 170. 交易逆序对总数（困难，原剑指 Offer 51）

**题目**：若 `i < j` 且 `nums[i] > nums[j]`，则称 `(i, j)` 是一个逆序对。求逆序对总数。

**思路**：

固定右端 j，逆序对就是「j 左边有多少个数比 `nums[j]` 大」。从左往右扫，树状数组记录已出现过
的数的**值域计数**：

- `processed` 是已插入总数；`prefix(rank(x))` 是已插入且 `<= x` 的个数；
- 两者相减就是「已插入且 `> x`」的个数，即左边比 x 大的个数，累加即可。

离散化把任意大小的数映射成 1..m 的排名，树状数组只需开 m 大小。

**代码**：

```python
def reverse_pairs_count(record):
    n = len(record)
    if n == 0:
        return 0
    sorted_vals = sorted(set(record))
    rank = {v: i + 1 for i, v in enumerate(sorted_vals)}
    bit = Fenwick(len(sorted_vals))

    ans = 0
    processed = 0
    for x in record:
        r = rank[x]
        ans += processed - bit.prefix(r)
        bit.add(r, 1)
        processed += 1
    return ans
```

```cpp
long long reversePairsCount(std::vector<int> record) {
    std::vector<int> vals = record;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    int processed = 0;
    for (int x : record) {
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), x) - vals.begin()) +
                1;
        ans += processed - bit.prefix(r);
        bit.add(r, 1);
        ++processed;
    }
    return ans;
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：`prefix(r)` 包含「等于 x」的元素，所以「大于」要用 `processed - prefix(r)`；
  若问「大于等于」才用 `prefix(r - 1)`。相等元素不算逆序对，别把它们误算进去。
- **相似题**：315 计算右侧小于当前元素的个数（把扫描方向改成从右往左）；493 翻转对（阈值换成
  `2x`）；1649 通过指令创建有序数组（每个位置取两侧较小者）；这一组都是「值域计数」的同一
  模板。

### 1649. 通过指令创建有序数组（困难）

**题目**：从左到右把 `instructions` 依次插入有序数组，每次插入代价 = 已有元素中「比它小的个数」
与「比它大的个数」的较小值。求代价之和对 1e9+7 取模的结果。

**思路**：

插入第 i 个数时，数组中恰好是前 i 个数，所以「比 x 小 / 大」就是「已出现过且 < x / > x」：

- `less = prefix(x - 1)`；
- `greater = processed - prefix(x)`（`prefix(x)` 含等于 x 的，等于既不算小也不算大）。

取 `min` 累加，再把 x 插入。本题值域是 `1..10^5`，可以直接开满；值域更大时照样先离散化。

**代码**：

```python
def create_sorted_array(instructions):
    MOD = 10 ** 9 + 7
    MAXV = 100000
    bit = Fenwick(MAXV)

    ans = 0
    processed = 0
    for x in instructions:
        less = bit.prefix(x - 1)
        greater = processed - bit.prefix(x)
        ans = (ans + min(less, greater)) % MOD
        bit.add(x, 1)
        processed += 1
    return ans
```

```cpp
int createSortedArray(std::vector<int> instructions) {
    const long long MOD = 1000000007LL;
    const int MAXV = 100000;

    Fenwick bit(MAXV);
    long long ans = 0;
    int processed = 0;
    for (int x : instructions) {
        long long less = bit.prefix(x - 1);
        long long greater = processed - bit.prefix(x);
        ans = (ans + std::min(less, greater)) % MOD;
        bit.add(x, 1);
        ++processed;
    }
    return static_cast<int>(ans);
}
```

- **复杂度**：时间 O(n log M)（M = 值域上界），空间 O(M)。
- **易错点**：等于 x 的个数既不属于 less 也不属于 greater，`prefix(x) - prefix(x-1)` 就是重复
  元素个数，天生被排除；`ans` 要按 `min` 累加后再取模，别最后才取（中间会很大）。
- **相似题**：LCR 170 逆序对（同模板，只是查询方向不同）；315 计算右侧小于当前元素的个数
  （另一侧视角）。

### 315. 计算右侧小于当前元素的个数（困难）

**题目**：返回数组 `counts`，`counts[i]` 是 `nums[i]` 右侧严格小于它的元素个数。

**思路**：

把扫描方向改成**从右往左**：这时「已经插入树状数组的数」恰好就是「当前元素的右侧元素」。
对每个位置先查 `prefix(rank - 1)`（已插入且 `< x` 的个数），记进答案，再把 x 插进去。

同样的动作，方向一换，就得到每个位置各自的统计量，而不是总和。

**代码**：

```python
def count_smaller(nums):
    n = len(nums)
    if n == 0:
        return []
    sorted_vals = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_vals)}
    bit = Fenwick(len(sorted_vals))

    ans = [0] * n
    for i in range(n - 1, -1, -1):
        r = rank[nums[i]]
        ans[i] = bit.prefix(r - 1)
        bit.add(r, 1)
    return ans
```

```cpp
std::vector<int> countSmaller(std::vector<int> nums) {
    std::vector<int> vals = nums;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    int n = static_cast<int>(nums.size());
    std::vector<int> ans(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), nums[i]) - vals.begin()) +
                1;
        ans[i] = static_cast<int>(bit.prefix(r - 1));
        bit.add(r, 1);
    }
    return ans;
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：题目要「严格小于」，所以查 `prefix(r - 1)`；从右往左写循环，`range(n-1, -1, -1)`
  别写成 `range(n-1, 0, -1)`（会漏掉下标 0）。
- **相似题**：LCR 170 逆序对（把答案求和而不是逐位保存）；493 翻转对（阈值换 `2x`）。

### 493. 翻转对（困难）

**题目**：若 `i < j` 且 `nums[i] > 2 * nums[j]`，称 `(i, j)` 为翻转对。返回翻转对数量。

**思路**：

仍是「从左扫、数前面比我大的」，只是比较阈值从 `nums[j]` 换成 `2 * nums[j]`：扫到 x 时，
答案是 `processed - prefix(rank(2x))`。

与逆序对的唯一区别是**离散化要多收一批值**：`2 * x` 也要参与排序去重，否则阈值 `2x` 落不到
某个排名上。另外数字可能为负、`2x` 可能溢出 32 位，C++ 要用 `long long`。

**代码**：

```python
def reverse_pairs(nums):
    n = len(nums)
    if n == 0:
        return 0
    vals = sorted(set(nums) | {2 * x for x in nums})
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    processed = 0
    for x in nums:
        ans += processed - bit.prefix(rank[2 * x])
        bit.add(rank[x], 1)
        processed += 1
    return ans
```

```cpp
int reversePairs(std::vector<int> nums) {
    std::vector<long long> vals;
    vals.reserve(nums.size() * 2);
    for (int x : nums) {
        vals.push_back(x);
        vals.push_back(2LL * x);
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    int processed = 0;
    for (int x : nums) {
        long long twox = 2LL * x;
        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), twox) - vals.begin()) +
                1;
        ans += processed - bit.prefix(r);

        int rx = static_cast<int>(
                     std::lower_bound(vals.begin(), vals.end(), static_cast<long long>(x)) -
                     vals.begin()) +
                 1;
        bit.add(rx, 1);
        ++processed;
    }
    return static_cast<int>(ans);
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：`2x` 一定要进离散化集合；`prefix(rank(2x))` 数的是 `<= 2x`，要「严格大于」所以
  用总数减它。负数参与比较时同理，别漏。
- **相似题**：LCR 170 逆序对（把 `2x` 换回 `x`）；315 计算右侧小于当前元素的个数；493 也用
  归并排序「合并时统计跨界对」可做，对照见 `docs/12-divide-conquer.md`。

---

## 模式三：前后各数一遍，乘法原理合并

**适用信号**：数**三元组** `i < j < k`，要求左右两侧对中间元素各有一项约束。

**核心动作**：按「枚举中间人 j」拆开——左半的某种计数 × 右半的某种计数；用树状数组求出一侧，
另一侧用「全局个数 − 已统计的一侧」推出。

### 1395. 统计作战单位数（中等）

**题目**：n 个士兵的能力值 `rating`，选出 `i < j < k` 使能力值严格递增或严格递减，求方案数。

**思路**：

以中间人 j 为中心，把两种队形写成乘积：

- 递增：左边比 j 小的个数 × 右边比 j 大的个数；
- 递减：左边比 j 大的个数 × 右边比 j 小的个数。

从左往右扫，用值域树状数组维护左侧元素，得到 `left_less`、`left_greater`；右侧个数用
「全局个数 − 左侧个数」推出：先统计每个值的总频次，再前缀求和得到「全局严格小于 rating[j]」
与「全局严格大于 rating[j]」的个数，各减去左侧的那部分即可。

**代码**：

```python
def num_teams(rating):
    maxv = max(rating)
    total = [0] * (maxv + 2)
    for x in rating:
        total[x] += 1

    less_total = [0] * (maxv + 2)
    greater_total = [0] * (maxv + 2)
    run = 0
    for v in range(1, maxv + 1):
        less_total[v] = run
        run += total[v]
    run = 0
    for v in range(maxv, 0, -1):
        greater_total[v] = run
        run += total[v]

    bit = Fenwick(maxv)
    ans = 0
    left_count = 0
    for x in rating:
        left_less = bit.prefix(x - 1)
        left_greater = left_count - bit.prefix(x)
        right_less = less_total[x] - left_less
        right_greater = greater_total[x] - left_greater
        ans += left_less * right_greater + left_greater * right_less
        bit.add(x, 1)
        left_count += 1
    return ans
```

```cpp
int numTeams(std::vector<int> rating) {
    int maxv = 0;
    for (int x : rating) maxv = std::max(maxv, x);

    std::vector<int> total(maxv + 2, 0);
    for (int x : rating) ++total[x];

    std::vector<int> lessTotal(maxv + 2, 0), greaterTotal(maxv + 2, 0);
    int run = 0;
    for (int v = 1; v <= maxv; ++v) {
        lessTotal[v] = run;
        run += total[v];
    }
    run = 0;
    for (int v = maxv; v >= 1; --v) {
        greaterTotal[v] = run;
        run += total[v];
    }

    Fenwick bit(maxv);
    long long ans = 0;
    int leftCount = 0;
    for (int x : rating) {
        long long leftLess = bit.prefix(x - 1);
        long long leftGreater = leftCount - bit.prefix(x);
        long long rightLess = lessTotal[x] - leftLess;
        long long rightGreater = greaterTotal[x] - leftGreater;
        ans += leftLess * rightGreater + leftGreater * rightLess;
        bit.add(x, 1);
        ++leftCount;
    }
    return static_cast<int>(ans);
}
```

- **复杂度**：时间 O(n log M + n + M)，空间 O(M)。
- **易错点**：`left_greater = left_count - prefix(x)`（不是 `left_count - prefix(x-1)`），因为
  `prefix(x)` 已把等于 x 的排除在「大于」之外；先乘后加，注意用 64 位（C++ 里 `ans` 是
  `long long`）。
- **相似题**：2179 统计数组中好三元组数目（同样是「枚举中间人 + 左小右大」）。

### 2179. 统计数组中好三元组数目（困难）

**题目**：给定两个 `0..n-1` 的排列 `nums1`、`nums2`，若下标三元组 `i < j < k` 在两个排列中
对应的三个数都按同样顺序出现，则称其为好三元组，求个数。

**思路**：

以 `nums1` 为基准，记 `pos[v]` 为 v 在 `nums1` 中的下标；按 `nums2` 的顺序取出这些下标，得到
数组 `b`。此时「在 `nums2` 中按顺序」已由下标天然满足，问题变成**数 `b` 的递增三元组**：

- 对每个 j，答案 += （b 左边比 `b[j]` 小的个数）×（b 右边比 `b[j]` 大的个数）。

左边更小用树状数组从左往右求；因为 `b` 是 `0..n-1` 的排列，「全局比 x 大的个数」就是
`n - 1 - x`，减去左边更大的个数即得右边更大的个数。

**代码**：

```python
def good_triplets(nums1, nums2):
    n = len(nums1)
    pos = [0] * n
    for i, v in enumerate(nums1):
        pos[v] = i
    b = [pos[v] for v in nums2]

    bit = Fenwick(n)
    ans = 0
    left_count = 0
    for x in b:
        left_less = bit.prefix(x)
        left_greater = left_count - bit.prefix(x + 1)
        right_greater = (n - 1 - x) - left_greater
        ans += left_less * right_greater
        bit.add(x + 1, 1)
        left_count += 1
    return ans
```

```cpp
long long goodTriplets(std::vector<int> nums1, std::vector<int> nums2) {
    int n = static_cast<int>(nums1.size());
    std::vector<int> pos(n, 0);
    for (int i = 0; i < n; ++i) pos[nums1[i]] = i;

    Fenwick bit(n);
    long long ans = 0;
    int leftCount = 0;
    for (int v : nums2) {
        int x = pos[v];
        long long leftLess = bit.prefix(x);
        long long leftGreater = leftCount - bit.prefix(x + 1);
        long long rightGreater = static_cast<long long>(n - 1 - x) - leftGreater;
        ans += leftLess * rightGreater;
        bit.add(x + 1, 1);
        ++leftCount;
    }
    return ans;
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：值为 `0..n-1`，树的排名是 `x+1`：`prefix(x)` 数的是 `< x`，`prefix(x+1)` 数的是
  `<= x`，两个别混；答案可能超 int，C++ 返回 `long long`。
- **相似题**：1395 统计作战单位数（同一套「中间人乘法」）；也可用「按值域扫 + 双树状数组」
  求解，本质一致。

---

## 模式四：前缀和 + 树状数组

**适用信号**：统计满足某个**区间和**条件的子数组个数。

**核心动作**：先把子数组和翻译成两个**前缀和之差**，条件随之变成「某个前缀落在某个区间里」，
再用树状数组对前缀和做值域计数。区间左闭右闭时，用两次二分定位排名再前缀和相减。

### 327. 区间和的个数（困难）

**题目**：求满足 `lower <= 区间和 <= upper` 的区间 `[i, j]` 的个数。

**思路**：

记 `prefix[k] = nums[0] + ... + nums[k-1]`，则区间和 `sum(i, j) = prefix[j+1] - prefix[i]`，
条件整理为：

```
prefix[j] - upper <= prefix[i] <= prefix[j] - lower
```

从左往右扫描前缀（先把 `prefix[0] = 0` 插进去，代表从下标 0 开始），扫到 `prefix[j]` 时查询
落在 `[prefix[j]-upper, prefix[j]-lower]` 内的已出现前缀个数，就是以 j 结尾的合法区间数。

因为要按**值**比较，先把所有前缀和排序去重，再用两次二分：`< lo` 的个数作为左界，
`<= hi` 的个数作为右界，两者在树状数组上的前缀和相减即区间内计数。

**代码**：

```python
def count_range_sum(nums, lower, upper):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    vals = sorted(set(prefix))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    bit.add(rank[prefix[0]], 1)
    for j in range(1, n + 1):
        lo = prefix[j] - upper
        hi = prefix[j] - lower
        left = bisect_left(vals, lo)
        right = bisect_right(vals, hi)
        ans += bit.prefix(right) - bit.prefix(left)
        bit.add(rank[prefix[j]], 1)
    return ans
```

```cpp
int countRangeSum(std::vector<int> nums, int lower, int upper) {
    int n = static_cast<int>(nums.size());
    std::vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + nums[i];

    std::vector<long long> vals = prefix;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    auto rank = [&](long long v) {
        return static_cast<int>(std::lower_bound(vals.begin(), vals.end(), v) -
                                vals.begin()) +
               1;
    };

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    bit.add(rank(prefix[0]), 1);
    for (int j = 1; j <= n; ++j) {
        long long lo = prefix[j] - upper;
        long long hi = prefix[j] - lower;
        int left = static_cast<int>(std::lower_bound(vals.begin(), vals.end(), lo) -
                                    vals.begin());
        int right = static_cast<int>(std::upper_bound(vals.begin(), vals.end(), hi) -
                                     vals.begin());
        ans += bit.prefix(right) - bit.prefix(left);
        bit.add(rank(prefix[j]), 1);
    }
    return static_cast<int>(ans);
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：别忘了先插入 `prefix[0] = 0`（否则所有以 0 结尾的区间会漏）；查询区间是闭区间，
  左界用 `lower_bound(lo)`（严格小于 lo 的个数），右界用 `upper_bound(hi)`（小于等于 hi 的
  个数），两者都表示「树状数组里的排名上界」；前缀和可能重复，树状数组按排名计数没问题，但
  排名映射要「值 → 唯一排名」。
- **相似题**：560 和为 K 的子数组（哈希表版，见 `docs/02-hash.md`）；974 和可被 K 整除的
  子数组（同余前缀和，见 `docs/04-prefix-sum.md`）；2426 满足不等式的数对数目（单边阈值版）。

### 2426. 满足不等式的数对数目（困难）

**题目**：求满足 `i < j` 且 `nums1[i] - nums2[i] <= nums1[j] - nums2[j] + k` 的下标对数目。

**思路**：

令 `diff[m] = nums1[m] - nums2[m]`，不等式变成 `diff[i] <= diff[j] + k`。于是对每个 j，只要
数「前面有多少个 diff 值不超过 `diff[j] + k`」。从左往右扫，树状数组维护已出现的 diff 计数；
阈值 `diff[j] + k` 用二分定位排名，直接前缀和。

这和 327 是同一副面孔——都是「差值 + 阈值计数」，区别只是阈值是单边（本题）还是双边（327）。

**代码**：

```python
def number_of_pairs(nums1, nums2, k):
    n = len(nums1)
    diff = [nums1[i] - nums2[i] for i in range(n)]

    vals = sorted(set(diff))
    rank = {v: i + 1 for i, v in enumerate(vals)}
    bit = Fenwick(len(vals))

    ans = 0
    for j in range(n):
        threshold = diff[j] + k
        idx = bisect_right(vals, threshold)
        ans += bit.prefix(idx)
        bit.add(rank[diff[j]], 1)
    return ans
```

```cpp
long long numberOfPairs(std::vector<int> nums1, std::vector<int> nums2, int k) {
    int n = static_cast<int>(nums1.size());
    std::vector<long long> diff(n);
    for (int i = 0; i < n; ++i)
        diff[i] = static_cast<long long>(nums1[i]) - nums2[i];

    std::vector<long long> vals = diff;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    Fenwick bit(static_cast<int>(vals.size()));
    long long ans = 0;
    for (int j = 0; j < n; ++j) {
        long long threshold = diff[j] + k;
        int idx = static_cast<int>(
            std::upper_bound(vals.begin(), vals.end(), threshold) - vals.begin());
        ans += bit.prefix(idx);

        int r = static_cast<int>(
                    std::lower_bound(vals.begin(), vals.end(), diff[j]) - vals.begin()) +
                1;
        bit.add(r, 1);
    }
    return ans;
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：`<=` 是闭的，阈值定位用 `upper_bound`（个数 = 小于等于阈值），别用
  `lower_bound`；先查再插，保证只统计 `i < j`。
- **相似题**：327 区间和的个数（双边阈值）；2426 也可以归并排序统计，思路与 493 类似。

---

## 模式五：树状数组维护「动态位置」

**适用信号**：元素会**移动**（被挪到最前 / 最后），但要快速知道它当前的**下标**。

**核心动作**：不要真的搬动元素，而是把它看成「占用某个槽位」。树状数组标记每个槽是否被占用，
「某元素当前的下标」= 它槽位左边已占用槽的数量（一段前缀和）。移动 = 换一个空槽。

### 1409. 查询带键的排列（中等）

**题目**：初始排列 `P = [1, 2, ..., m]`。对每个查询 q：返回 q 在 P 中的下标，然后把 q 移到 P
的开头。返回所有下标。

**思路**：

朴素做法每次移动 O(m)。这里用「预留空槽」技巧：开一个大小 `m + n` 的空间，把初始的 1..m 放在
靠后的 `n+1 .. n+m` 槽，前面 n 个槽留给「移到开头」用。树状数组里每个**已占用**的槽记 1，于是：

- 元素当前下标 = `prefix(槽位 - 1)`（左边被占用的槽数 = 前面有几个元素）；
- 移到开头：原槽位减 1，前面的下一个空槽加 1，并更新该元素的槽位。

每次查询占用一个空槽，n 次查询正好用掉预留的 n 个槽。

**代码**：

```python
def process_queries(queries, m):
    n = len(queries)
    bit = Fenwick(m + n)

    pos = [0] * (m + 1)
    for v in range(1, m + 1):
        pos[v] = n + v
        bit.add(pos[v], 1)

    ans = []
    next_pos = n
    for q in queries:
        p = pos[q]
        ans.append(bit.prefix(p - 1))
        bit.add(p, -1)
        bit.add(next_pos, 1)
        pos[q] = next_pos
        next_pos -= 1
    return ans
```

```cpp
std::vector<int> processQueries(std::vector<int> queries, int m) {
    int n = static_cast<int>(queries.size());
    Fenwick bit(m + n);

    std::vector<int> pos(m + 1, 0);
    for (int v = 1; v <= m; ++v) {
        pos[v] = n + v;
        bit.add(pos[v], 1);
    }

    std::vector<int> ans;
    ans.reserve(n);
    int nextPos = n;
    for (int q : queries) {
        int p = pos[q];
        ans.push_back(static_cast<int>(bit.prefix(p - 1)));
        bit.add(p, -1);
        bit.add(nextPos, 1);
        pos[q] = nextPos;
        --nextPos;
    }
    return ans;
}
```

- **复杂度**：时间 O((m + n) log(m + n))，空间 O(m + n)。
- **易错点**：空间要开 `m + n`（每次查询腾一个空槽）；移动时先「原槽 -1」再「新槽 +1」，
  顺序不影响结果但要成对；`pos` 记录的是元素的**槽位**而不是下标，两者靠前缀和换算。
- **相似题**：146 LRU 缓存（另一种「动态顺序」结构，见 `docs/19-design.md`）；981 基于时间的
  键值存储（时间轴上的动态版本，见 `docs/19-design.md`）。

---

## 规律总结

1. **树状数组解决两类问题**：①数组会变、要查区间和（模式一）；②扫描过程中要数「已出现过
   且满足某种大小关系的数」（模式二~四）。看到这两种信号，先想它。

2. **两个走法背下来**：查询前缀和 `i -= lowbit(i)`，单点修改 `i += lowbit(i)`。它是「分段和」
   的逆过程，不重不漏；下标一律从 1 开始。

3. **值域计数是它的主战场**。把「数值大小」当坐标，边扫边插，就能回答「左边有多少个比 x 大 /
   小」。数值范围大或有负数时，一律先**离散化**成排名。

4. **离散化的对象要收全**。493 的阈值是 `2x`，所以 `2x` 也要进集合；只收原数组会漏掉阈值。
   比较产生的任何中间量，只要会拿去查询，就得能被定位到排名。

5. **「严格」和「非严格」只差一个下标**：严格小于查 `prefix(r - 1)`，小于等于查 `prefix(r)`；
   严格大于用 `总数 - prefix(r)`，大于等于用 `总数 - prefix(r - 1)`。写之前先把不等式写成
   文字，再选边界。

6. **三元组先想「枚举中间人」**：中间元素的约束常能拆成「左边某计数 × 右边某计数」，用一次
   从左扫描拿到左侧、用「全局 − 左侧」拿到右侧，避免两次扫描。

7. **区间和先转成前缀和之差**。`sum(i, j) = prefix[j+1] - prefix[i]` 一写出来，区间计数就变成
   「前缀落在一个区间里」的值域计数——这是 327 的全部秘密。别忘了插入 `prefix[0] = 0`。

8. **查询区间用两次二分**：`lower_bound(lo)` 给严格小于 lo 的个数（左界），`upper_bound(hi)`
   给小于等于 hi 的个数（右界），再在树状数组上做前缀和相减。

9. **树状数组也能描述「顺序」**：元素会移动时，把位置当槽位、用「占用标记」的树状数组求当前
   下标（1409），比真的搬动元素高效得多。

10. **复杂度几乎总是 O(n log n)**，空间 O(n)（值域树状数组则 O(M)）。当题目值域很小（如
    1..n 的排列、1..10^5），可以省掉离散化直接开满；否则老老实实排序去重。

11. **和归并排序的关系**：逆序对、翻转对、数对计数都能用「归并时统计跨界对」在 O(n log n)
    内完成（见 `docs/12-divide-conquer.md`）。树状数组和它实现不同、复杂度相同：树状数组写法
    更短、更好记，归并排序则不需要离散化。两把工具都值得会。
