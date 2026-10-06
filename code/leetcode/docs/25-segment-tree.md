# 线段树：任意可合并的区间信息，都能压到 log n

树状数组用 `lowbit` 把「单点修改 + 前缀和」压到 O(log n)，是性价比最高的区间结构，但它有两个
天花板：只擅长「和」这类**可差分**（能由前缀相减得到）的信息，而且不会做**区间修改**（给一整段
同时加一个数）。一旦题目要求「区间最大值」「区间赋值」「区间翻转」「求区间里最长的同类连续段」，
树状数组就无能为力了。

**线段树**就是那个通用的答案。它把数组摊成一棵完全二叉树：叶子是单个元素，内部节点维护「左右
两个孩子的合并结果」。只要合并运算满足结合律（和、最小值、最大值、gcd、最长连续段……），线段树
都能在 O(log n) 内完成区间查询。再加上**懒标记（lazy）**，连「把一整段区间 +1 / 赋值 / 翻转」
这样的区间修改也能做到 O(log n)。

可以这么选型：

- 只是单点改、查前缀和/前缀计数 → **树状数组**，常数小、代码短（见第 24 篇）。
- 要查区间最值/求区间里某种可合并信息，或有区间修改 → **线段树**。
- 坐标大得开不下数组 → **动态开点线段树**，节点用到才创建。

## 线段树的三个构件

写线段树时永远只问三个问题：

1. **合并（pull）**：两个孩子合并成父亲，算什么？比如和是相加，最大值取 max，
   最长连续段要额外看左右边界字符。
2. **查询（query）**：目标区间与当前节点区间是什么关系？
   - 完全不相交 → 返回单位元（和是 0，最大值是 -inf）；
   - 完全被包含 → 直接返回本节点存的信息；
   - 部分相交 → 递归两个孩子再合并。
3. **懒标记（lazy）**：整段被修改时先「记账」不往下传，等真正要访问孩子时再**下推（push）**。
   加标记的下推是把增量加到孩子；赋值标记的下推是直接覆盖孩子。

后面每道题，无外乎在这三个问题上做不同选择。

## 本篇要解决的问题与模式

| 模式 | 线段树扮演的角色 | 题目 | 难度 |
|---|---|---|---|
| 模式一：单点修改 + 区间查询 | 最基础模板，维护「和」或「最大值」 | 307. 区域和检索 - 数组可修改 / 2407. 最长递增子序列 II | 中等/困难 |
| 模式二：动态开点 + 区间加 + 区间最大 | 坐标巨大，维护覆盖次数 | 729. 我的日程安排表 I / 731. 我的日程安排表 II / 732. 我的日程安排表 III | 中等/困难 |
| 模式三：区间赋值 | 懒标记是「覆盖」而不是「增量」 | 715. Range 模块 / 699. 掉落的方块 | 困难 |
| 模式四：区间翻转 + 区间和 | 0/1 翻转后和变「长度 − 和」 | 2569. 更新数组后处理求和查询 | 困难 |
| 模式五：区间合并信息 | 节点存「左前缀/右后缀/最优」 | 2213. 由单个字符重复的最长子字符串 | 困难 |
| 模式六：线段树上二分 | 在树上找第一个满足阈值的位置 | 2940. 找到 Alice 和 Bob 可以相遇的建筑 | 困难 |

一句话记住三组关系：**「和」看可差分、「最值」看可合并、「懒标记」看区间修改的种类。**

---

## 模式一：单点修改 + 区间查询

**适用信号**：数组会被单点修改，同时要频繁查询某个区间上的信息。

**核心动作**：叶子存单点，父亲存合并值；修改只走一条到根的路径，查询只拆成 O(log n) 个整块。

### 307. 区域和检索 - 数组可修改（中等）

**题目**：实现 `update(i, val)`（把 `nums[i]` 改成 val）和 `sumRange(l, r)`（返回闭区间和）。

**思路**：

静态区间和用前缀和 O(1) 查询，但一改就要重算后缀；线段树让修改和查询都是 O(log n)。这里用
**自底向上的迭代写法**：开一个长度 `2n` 的数组，叶子放在下标 `[n, 2n)`，父亲 `i` 的两个孩子是
`2i`、`2i+1`。建树从 `n-1` 倒着往上把和算出来。

单点修改：改掉叶子后 `i //= 2` 一路向上重算。区间查询把左闭右开的 `[l, r)` 映射成叶子下标
`[l+n, r+n)`，用两个游标向中间夹：**左游标是奇数**说明它对应的整块不能被父节点代表，必须单独
取走并右移一格；**右游标是奇数**同理要先左移一格再取。然后两个游标一起 `// 2` 上跳，直到相遇。
这样恰好取出不重不漏的 O(log n) 个整块。

**代码**：

Python：

```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        for i, v in enumerate(nums):
            self.tree[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index, val):
        i = index + self.n
        self.tree[i] = val
        i //= 2
        while i:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def sumRange(self, left, right):
        res = 0
        l, r = left + self.n, right + self.n + 1
        while l < r:
            if l & 1:
                res += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.tree[r]
            l //= 2
            r //= 2
        return res
```

C++：

```cpp
class NumArray {
    int n;
    std::vector<long long> tree;

public:
    explicit NumArray(std::vector<int> nums) : n(static_cast<int>(nums.size())), tree(2 * n, 0) {
        for (int i = 0; i < n; ++i) tree[n + i] = nums[i];
        for (int i = n - 1; i > 0; --i) tree[i] = tree[2 * i] + tree[2 * i + 1];
    }

    void update(int index, int val) {
        int i = index + n;
        tree[i] = val;
        for (i /= 2; i > 0; i /= 2) tree[i] = tree[2 * i] + tree[2 * i + 1];
    }

    long long sumRange(int left, int right) {
        long long res = 0;
        for (int l = left + n, r = right + n + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res += tree[l++];
            if (r & 1) res += tree[--r];
        }
        return res;
    }
};
```

**复杂度**：建树 O(n)；`update`、`sumRange` 各 O(log n)；空间 O(n)。

**易错点**：

- 迭代线段树用「父亲 `i`，孩子 `2i`/`2i+1`」，除根以外下标从 1 起；数组长度开 `2n` 即可
  （`n=1` 也成立）。
- 查询区间是**左闭右开** `[left+n, right+n+1)`，右端要 +1；写错会出现差一错误。

**相似题**：线段树的「值域版」是 2407（下面这题）；如果只做单点改 + 前缀计数，用树状数组
（第 24 篇）更划算。

### 2407. 最长递增子序列 II（困难）

**题目**：给定数组 `nums` 和整数 `k`，求满足 `i < j`、`nums[i] < nums[j]` 且
`nums[j] - nums[i] <= k` 的最长子序列长度。

**思路**：

普通 LIS 是 `dp[j] = 1 + max{ dp[i] : i < j, nums[i] < nums[j] }`。加上「差值 <= k」后，转移是
「在**值域区间** `[nums[j]-k, nums[j]-1]` 里找最大的 dp」。于是开一棵**最大值线段树，下标就是
数值**：

- 处理到 `nums[j]` 时，先查这个区间的最大值 `best`，得到 `dp[j] = best + 1`；
- 再把 `dp[j]` 写到下标 `nums[j]` 上（单点取 max）。

为什么「边扫边查」天然满足 `i < j`：我们按数组顺序从左往右处理，写进树里的都是更早出现的元素，
时序约束已经内建进去了。这其实和树状数组数逆序对（第 24 篇）是同一个套路，只是把「值域计数」
换成了「值域最大值」。

**代码**：

Python：

```python
def length_of_lis(nums, k):
    max_v = max(nums)
    size = max_v + 1
    tree = [0] * (2 * size)

    def update(pos, val):
        i = pos + size
        if tree[i] >= val:
            return
        tree[i] = val
        i //= 2
        while i:
            tree[i] = max(tree[2 * i], tree[2 * i + 1])
            i //= 2

    def query(lo, hi):
        res = 0
        l, r = lo + size, hi + size + 1
        while l < r:
            if l & 1:
                res = max(res, tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, tree[r])
            l //= 2
            r //= 2
        return res

    ans = 0
    for v in nums:
        lo, hi = max(0, v - k), v - 1
        best = query(lo, hi) if lo <= hi else 0
        cur = best + 1
        update(v, cur)
        ans = max(ans, cur)
    return ans
```

C++：

```cpp
int lengthOfLIS(const std::vector<int>& nums, int k) {
    int maxV = *std::max_element(nums.begin(), nums.end());
    int size = maxV + 1;
    std::vector<int> tree(2 * size, 0);
    auto update = [&](int pos, int val) {
        int i = pos + size;
        if (tree[i] >= val) return;
        tree[i] = val;
        for (i /= 2; i > 0; i /= 2) tree[i] = std::max(tree[2 * i], tree[2 * i + 1]);
    };
    auto query = [&](int lo, int hi) {
        int res = 0;
        for (int l = lo + size, r = hi + size + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res = std::max(res, tree[l++]);
            if (r & 1) res = std::max(res, tree[--r]);
        }
        return res;
    };
    int ans = 0;
    for (int v : nums) {
        int lo = std::max(0, v - k), hi = v - 1;
        int best = (lo <= hi) ? query(lo, hi) : 0;
        int cur = best + 1;
        update(v, cur);
        ans = std::max(ans, cur);
    }
    return ans;
}
```

**复杂度**：O(n log V)，V 为值域上界（题目保证 `nums[i] <= 1e5`）；空间 O(V)。

**易错点**：

- 查询区间是 `[v-k, v-1]`：**严格递增**所以要排除等于 v 的位置；`v-k` 可能为负，要夹到 0。
- 区间为空（`lo > hi`，比如 `v=0` 或 `k=0` 且 v 很小）时不能查询，直接令 `best=0`。

**相似题**：求「最长递增子序列」的朴素/二分/DP 版本见第 13 篇（300）；「值域上数个数」见第 24 篇
（LCR 170 / 315）。

---

## 模式二：动态开点 + 区间加 + 区间最大值

**适用信号**：区间是「在线」给出的，坐标范围很大（到 1e9），要维护每个点被覆盖的次数。

**核心动作**：用**动态开点**——节点用到时才创建，单次操作只多出 O(log C) 个节点。区间加 +1，
查询区间最大值。日历三题共用这棵树。

Python 版线段树（729、731、732 三题逐字相同）：

```python
class SegTree:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.lc = [0, 0]
        self.rc = [0, 0]
        self.mx = [0, 0]
        self.lz = [0, 0]

    def _new(self):
        self.lc.append(0)
        self.rc.append(0)
        self.mx.append(0)
        self.lz.append(0)
        return len(self.mx) - 1

    def _apply(self, o, v):
        self.mx[o] += v
        self.lz[o] += v

    def _push(self, o, l, r):
        if l >= r:
            return
        if not self.lc[o]:
            self.lc[o] = self._new()
        if not self.rc[o]:
            self.rc[o] = self._new()
        v = self.lz[o]
        if v:
            self._apply(self.lc[o], v)
            self._apply(self.rc[o], v)
            self.lz[o] = 0

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, v)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._update(self.lc[o], l, m, ql, qr, v)
        if qr > m:
            self._update(self.rc[o], m + 1, r, ql, qr, v)
        self.mx[o] = max(self.mx[self.lc[o]], self.mx[self.rc[o]])

    def add(self, ql, qr, v):
        if ql <= qr:
            self._update(1, self.lo, self.hi, ql, qr, v)

    def _query(self, o, l, r, ql, qr):
        if not o:
            return 0
        if ql <= l and r <= qr:
            return self.mx[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res = max(res, self._query(self.lc[o], l, m, ql, qr))
        if qr > m:
            res = max(res, self._query(self.rc[o], m + 1, r, ql, qr))
        return res

    def query(self, ql, qr):
        if ql > qr:
            return 0
        return self._query(1, self.lo, self.hi, ql, qr)
```

C++ 版线段树（三题逐字相同）：

```cpp
struct SegTree {
    long long lo, hi;
    std::vector<int> lc, rc, mx, lz;
    SegTree(long long lo_, long long hi_)
        : lo(lo_), hi(hi_), lc(2, 0), rc(2, 0), mx(2, 0), lz(2, 0) {}

    int newNode() {
        lc.push_back(0);
        rc.push_back(0);
        mx.push_back(0);
        lz.push_back(0);
        return static_cast<int>(mx.size()) - 1;
    }
    void applyNode(int o, int v) {
        mx[o] += v;
        lz[o] += v;
    }
    void push(int o, long long l, long long r) {
        if (l >= r) return;
        if (!lc[o]) lc[o] = newNode();
        if (!rc[o]) rc[o] = newNode();
        if (lz[o]) {
            applyNode(lc[o], lz[o]);
            applyNode(rc[o], lz[o]);
            lz[o] = 0;
        }
    }
    void update(int o, long long l, long long r, long long ql, long long qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, v);
            return;
        }
        push(o, l, r);
        long long m = (l + r) / 2;
        if (ql <= m) update(lc[o], l, m, ql, qr, v);
        if (qr > m) update(rc[o], m + 1, r, ql, qr, v);
        mx[o] = std::max(mx[lc[o]], mx[rc[o]]);
    }
    int query(int o, long long l, long long r, long long ql, long long qr) {
        if (!o) return 0;
        if (ql <= l && r <= qr) return mx[o];
        push(o, l, r);
        long long m = (l + r) / 2;
        int res = 0;
        if (ql <= m) res = std::max(res, query(lc[o], l, m, ql, qr));
        if (qr > m) res = std::max(res, query(rc[o], m + 1, r, ql, qr));
        return res;
    }
    void add(long long ql, long long qr, int v) {
        if (ql <= qr) update(1, lo, hi, ql, qr, v);
    }
    int query(long long ql, long long qr) {
        if (ql > qr) return 0;
        return query(1, lo, hi, ql, qr);
    }
};
```

### 729. 我的日程安排表 I（中等）

**题目**：`book(start, end)` 判断半开区间 `[start, end)` 是否与已预订区间都不重叠；不重叠则加入
并返回 `True`，否则不加入返回 `False`。

**思路**：

两个半开区间相交，等价于「存在某个整数点被覆盖两次」。于是「会不会重叠」=「查询 `[start, end-1]`
上已有的最大覆盖次数是否 `>= 1`」。注意题目区间是半开的，`end` 这个点不属于区间，真正被占用的
整数点是 `start ... end-1`，所以查询/更新都用左闭右闭的 `[start, end-1]`。

**代码**：

Python：

```python
class MyCalendar:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        if self.tree.query(start, end - 1) >= 1:
            return False
        self.tree.add(start, end - 1, 1)
        return True
```

C++：

```cpp
class MyCalendar {
    SegTree tree;

public:
    MyCalendar() : tree(0, 1000000000LL) {}
    bool book(int start, int end) {
        if (tree.query(start, end - 1) >= 1) return false;
        tree.add(start, end - 1, 1);
        return true;
    }
};
```

**复杂度**：每次 `book` O(log C)，C = 1e9；空间 O(q log C)。

**易错点**：查询区间是 `[start, end-1]` 而不是 `[start, end]`，多占一个端点会把「首尾相接」
误判成重叠。

**相似题**：下面 731、732 是同一棵树的升级版。

### 731. 我的日程安排表 II（中等）

**题目**：允许同一时间点最多被两个日程覆盖；若新日程会造成「三重预订」则拒绝。

**思路**：

同一棵树，唯一区别是阈值从 1 变成 2：新日程会给覆盖到的每个点 +1，所以只要 `[start, end-1]`
上当前最大值已经 `>= 2`，加入后就会出现 `>= 3`，必须拒绝。对照 729：`>= 1` 拒绝任何重叠，
`>= 2` 只拒绝第三重。

**代码**：

Python：

```python
class MyCalendarTwo:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        if self.tree.query(start, end - 1) >= 2:
            return False
        self.tree.add(start, end - 1, 1)
        return True
```

C++：

```cpp
class MyCalendarTwo {
    SegTree tree;

public:
    MyCalendarTwo() : tree(0, 1000000000LL) {}
    bool book(int start, int end) {
        if (tree.query(start, end - 1) >= 2) return false;
        tree.add(start, end - 1, 1);
        return true;
    }
};
```

**复杂度**：每次 `book` O(log C) ；空间 O(q log C)。

**易错点**：判断的是「加入**后**会不会达到 3 重」，所以是查加入前的最大值是否已经 `>= 2`，
而不是 `>= 3`。

**相似题**：729、732。

### 732. 我的日程安排表 III（困难）

**题目**：每次加入日程后，返回「同时进行的日程数的最大值」（最大重叠数）。

**思路**：

不用拒绝，直接返回加入后的全局最大重叠数。线段树的**根节点维护的就是全局最大值**，所以区间
`[start, end-1]` 整体 +1 之后读 `tree.mx[1]` 即可。729/731 是「加入前查阈值」，732 是「加入后
读峰值」，是同一棵树的三种用法。

**代码**：

Python：

```python
class MyCalendarThree:
    def __init__(self):
        self.tree = SegTree(0, 10 ** 9)

    def book(self, start, end):
        self.tree.add(start, end - 1, 1)
        return self.tree.mx[1]
```

C++：

```cpp
class MyCalendarThree {
    SegTree tree;

public:
    MyCalendarThree() : tree(0, 1000000000LL) {}
    int book(int start, int end) {
        tree.add(start, end - 1, 1);
        return tree.mx[1];
    }
};
```

**复杂度**：每次 `book` O(log C)；空间 O(q log C)。

**易错点**：根节点下标固定是 1（`_new` 从下标 2 开始分配），别读成 `mx[0]`。

**相似题**：729、731。

---

## 模式三：区间赋值

**适用信号**：操作是「把一整段设成某个值」，而不是累加。

**核心动作**：懒标记存「要赋的值」（这里用 1/0 或 -1 表示「无标记」），下推时**覆盖**孩子的
值和标记。因为赋值会抹掉历史，不能像加法那样累加。

### 715. Range 模块（困难）

**题目**：维护半开区间 `[left, right)` 是否被「跟踪」，支持 `addRange`、`removeRange`、
`queryRange`（整段是否都被跟踪）。

**思路**：

每个整数点只有「跟踪 / 不跟踪」两态，三种操作都是「区间整体赋 1 / 赋 0」，查询是「区间里被跟踪
的点数是否等于区间长度」。节点维护 `cnt` = 该区间被跟踪的整数点个数；赋值时 `cnt` 直接等于
（值 × 区间长度）。懒标记 `lz` 取 `-1` 表示无标记，取 0/1 表示整段被赋成该值。

坐标到 1e9，用动态开点。`_push` 里**必须无条件创建孩子**再判断是否有标记下推——否则遇到
「无标记但还没有孩子」的节点时会递归到空节点 0，把数据写到无效下标上。

**代码**：

Python：

```python
class RangeModule:
    def __init__(self):
        self.lo, self.hi = 1, 10 ** 9
        self.lc = [0, 0]
        self.rc = [0, 0]
        self.cnt = [0, 0]
        self.lz = [-1, -1]

    def _new(self):
        self.lc.append(0)
        self.rc.append(0)
        self.cnt.append(0)
        self.lz.append(-1)
        return len(self.cnt) - 1

    def _apply(self, o, l, r, v):
        self.cnt[o] = v * (r - l + 1)
        self.lz[o] = v

    def _push(self, o, l, r):
        if l >= r:
            return
        if not self.lc[o]:
            self.lc[o] = self._new()
        if not self.rc[o]:
            self.rc[o] = self._new()
        v = self.lz[o]
        if v != -1:
            m = (l + r) // 2
            self._apply(self.lc[o], l, m, v)
            self._apply(self.rc[o], m + 1, r, v)
            self.lz[o] = -1

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, l, r, v)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._update(self.lc[o], l, m, ql, qr, v)
        if qr > m:
            self._update(self.rc[o], m + 1, r, ql, qr, v)
        self.cnt[o] = self.cnt[self.lc[o]] + self.cnt[self.rc[o]]

    def _query(self, o, l, r, ql, qr):
        if not o:
            return 0
        if ql <= l and r <= qr:
            return self.cnt[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res += self._query(self.lc[o], l, m, ql, qr)
        if qr > m:
            res += self._query(self.rc[o], m + 1, r, ql, qr)
        return res

    def addRange(self, left, right):
        self._update(1, self.lo, self.hi, left, right - 1, 1)

    def removeRange(self, left, right):
        self._update(1, self.lo, self.hi, left, right - 1, 0)

    def queryRange(self, left, right):
        return self._query(1, self.lo, self.hi, left, right - 1) == right - left
```

C++：

```cpp
struct SegTree {
    long long lo, hi;
    std::vector<int> lc, rc, lz;
    std::vector<long long> cnt;
    SegTree(long long lo_, long long hi_)
        : lo(lo_), hi(hi_), lc(2, 0), rc(2, 0), lz(2, -1), cnt(2, 0) {}

    int newNode() {
        lc.push_back(0);
        rc.push_back(0);
        lz.push_back(-1);
        cnt.push_back(0);
        return static_cast<int>(cnt.size()) - 1;
    }
    void applyNode(int o, long long l, long long r, int v) {
        cnt[o] = static_cast<long long>(v) * (r - l + 1);
        lz[o] = v;
    }
    void push(int o, long long l, long long r) {
        if (l >= r) return;
        if (!lc[o]) lc[o] = newNode();
        if (!rc[o]) rc[o] = newNode();
        if (lz[o] != -1) {
            long long m = (l + r) / 2;
            applyNode(lc[o], l, m, lz[o]);
            applyNode(rc[o], m + 1, r, lz[o]);
            lz[o] = -1;
        }
    }
    void update(int o, long long l, long long r, long long ql, long long qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, l, r, v);
            return;
        }
        push(o, l, r);
        long long m = (l + r) / 2;
        if (ql <= m) update(lc[o], l, m, ql, qr, v);
        if (qr > m) update(rc[o], m + 1, r, ql, qr, v);
        cnt[o] = cnt[lc[o]] + cnt[rc[o]];
    }
    long long query(int o, long long l, long long r, long long ql, long long qr) {
        if (!o) return 0;
        if (ql <= l && r <= qr) return cnt[o];
        push(o, l, r);
        long long m = (l + r) / 2, res = 0;
        if (ql <= m) res += query(lc[o], l, m, ql, qr);
        if (qr > m) res += query(rc[o], m + 1, r, ql, qr);
        return res;
    }
    void assign(long long ql, long long qr, int v) { update(1, lo, hi, ql, qr, v); }
    bool full(long long ql, long long qr) { return query(1, lo, hi, ql, qr) == qr - ql + 1; }
};

class RangeModule {
    SegTree tree;

public:
    RangeModule() : tree(1, 1000000000LL) {}
    void addRange(int left, int right) { tree.assign(left, right - 1, 1); }
    void removeRange(int left, int right) { tree.assign(left, right - 1, 0); }
    bool queryRange(int left, int right) { return tree.full(left, right - 1); }
};
```

**复杂度**：每次操作 O(log C)，C = 1e9；空间 O(q log C)。

**易错点**：

- **动态开点下推时要先建孩子**，不能只在「有懒标记」时才建；否则递归会落到空节点。
- `lz` 用 `-1` 表示「无标记」，赋值 0 也是合法标记，不能用 0 当哨兵。
- `cnt` 用 64 位，`v * (r-l+1)` 在 1e9 量级不会溢出 int，但求和时保守用 `long long` 更稳。

**相似题**：699（下面这题）也是区间赋值，只是赋值的是高度。

### 699. 掉落的方块（困难）

**题目**：依次掉落方块，第 i 个占据 `[left_i, left_i + side_i)`，落在它下方已被占据的最大高度
之上；记录每次掉落后所有方块叠成的最高高度。

**思路**：

先把所有方块的左右端点收集起来排序去重，得到一串坐标点，相邻两点之间是高度恒定的区间，这样
把 1e9 的坐标压到至多 2n 个。对每个方块：

1. 查 `[left, left+side)` 区间内当前最大高度 `base`；
2. 它落在 `base + side`；
3. 把这个区间整体**赋成** `base + side`；
4. 维护全局最大高度 `cur`，记入答案。

为什么「赋值」够用：新高度 `base + side` 严格大于区间内所有旧高度，整块被新方块平整地抬高，
直接覆盖即可，不用再取 max。线段树支持区间赋值 + 区间最大值。

**代码**：

Python：

```python
class SegTree:
    def __init__(self, n):
        self.n = n
        self.mx = [0] * (4 * n)
        self.lz = [-1] * (4 * n)

    def _apply(self, o, v):
        self.mx[o] = v
        self.lz[o] = v

    def _push(self, o):
        if self.lz[o] != -1:
            self._apply(2 * o, self.lz[o])
            self._apply(2 * o + 1, self.lz[o])
            self.lz[o] = -1

    def _update(self, o, l, r, ql, qr, v):
        if ql <= l and r <= qr:
            self._apply(o, v)
            return
        self._push(o)
        m = (l + r) // 2
        if ql <= m:
            self._update(2 * o, l, m, ql, qr, v)
        if qr > m:
            self._update(2 * o + 1, m + 1, r, ql, qr, v)
        self.mx[o] = max(self.mx[2 * o], self.mx[2 * o + 1])

    def _query(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.mx[o]
        self._push(o)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res = max(res, self._query(2 * o, l, m, ql, qr))
        if qr > m:
            res = max(res, self._query(2 * o + 1, m + 1, r, ql, qr))
        return res

    def assign(self, ql, qr, v):
        self._update(1, 0, self.n - 1, ql, qr, v)

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)


def falling_squares(positions):
    xs = sorted({x for left, side in positions for x in (left, left + side)})
    idx = {x: i for i, x in enumerate(xs)}
    st = SegTree(len(xs) - 1)
    res = []
    cur = 0
    for left, side in positions:
        li = idx[left]
        ri = idx[left + side] - 1
        base = st.query(li, ri)
        h = base + side
        st.assign(li, ri, h)
        cur = max(cur, h)
        res.append(cur)
    return res
```

C++：

```cpp
struct SegTree {
    int n;
    std::vector<int> mx, lz;
    explicit SegTree(int n_) : n(n_), mx(4 * n_, 0), lz(4 * n_, -1) {}

    void applyNode(int o, int v) {
        mx[o] = v;
        lz[o] = v;
    }
    void push(int o) {
        if (lz[o] != -1) {
            applyNode(2 * o, lz[o]);
            applyNode(2 * o + 1, lz[o]);
            lz[o] = -1;
        }
    }
    void update(int o, int l, int r, int ql, int qr, int v) {
        if (ql <= l && r <= qr) {
            applyNode(o, v);
            return;
        }
        push(o);
        int m = (l + r) / 2;
        if (ql <= m) update(2 * o, l, m, ql, qr, v);
        if (qr > m) update(2 * o + 1, m + 1, r, ql, qr, v);
        mx[o] = std::max(mx[2 * o], mx[2 * o + 1]);
    }
    int query(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return mx[o];
        push(o);
        int m = (l + r) / 2, res = 0;
        if (ql <= m) res = std::max(res, query(2 * o, l, m, ql, qr));
        if (qr > m) res = std::max(res, query(2 * o + 1, m + 1, r, ql, qr));
        return res;
    }
    void assign(int ql, int qr, int v) { update(1, 0, n - 1, ql, qr, v); }
    int query(int ql, int qr) { return query(1, 0, n - 1, ql, qr); }
};

std::vector<int> fallingSquares(const std::vector<std::pair<int, int>>& positions) {
    std::vector<int> xs;
    for (auto [left, side] : positions) {
        xs.push_back(left);
        xs.push_back(left + side);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::map<int, int> idx;
    for (int i = 0; i < static_cast<int>(xs.size()); ++i) idx[xs[i]] = i;

    SegTree st(static_cast<int>(xs.size()) - 1);
    std::vector<int> res;
    int cur = 0;
    for (auto [left, side] : positions) {
        int li = idx[left];
        int ri = idx[left + side] - 1;
        int base = st.query(li, ri);
        int h = base + side;
        st.assign(li, ri, h);
        cur = std::max(cur, h);
        res.push_back(cur);
    }
    return res;
}
```

**复杂度**：坐标压缩 O(n log n)；每个方块线段树操作 O(log n)；总 O(n log n)，空间 O(n)。

**易错点**：

- 方块右边是 `left + side`（半开），区间下标是 `[idx[left], idx[left+side]-1]`，别写成
  `idx[left+side]`。
- 每步的答案要输出**历史最大高度**（`cur`），不是当前这个方块的高度。

**相似题**：715 的区间赋值；「闭区间差分」的离线做法见第 4 篇（1109 航班预订统计）。

---

## 模式四：区间翻转 + 区间和

**适用信号**：一段区间里的 0/1 全部取反，同时要维护区间里 1 的个数。

**核心动作**：翻转后「1 的个数」= 区间长度 − 原来的个数，懒标记是一个「是否翻转」的布尔量，
同一区间翻两次等于没翻（异或性质）。

### 2569. 更新数组后处理求和查询（困难）

**题目**：维护 `nums1`、`nums2` 和三类查询：`[1,l,r]` 把 `nums1[l..r]` 翻转；
`[2,p,0]` 令每个 `nums2[i] += nums1[i]*p`；`[3,0,0]` 返回 `sum(nums2)`。

**思路**：

分两步看：

- `nums1` 的翻转用一个支持「区间翻转」的线段树维护，节点存 1 的个数 `s`。翻转一个区间时
  `s = 区间长度 - s`，并把翻转标记取反下传。
- **关键观察**：我们从不访问 `nums2` 的单个元素，只关心它的总和。类型 2 会让总和增加
  `p * sum(nums1)`（按当前的 `nums1` 加权）；维护 `total = sum(nums2)`，类型 2 就
  `total += p * 当前 sum(nums1)`，类型 3 直接返回 `total`。之后 `nums1` 再怎么翻转，已经加进
  `nums2` 的历史值都不变，所以不必回改。

**代码**：

Python：

```python
class SegTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.s = [0] * (4 * self.n)
        self.lz = [False] * (4 * self.n)
        self._build(1, 0, self.n - 1, nums)

    def _build(self, o, l, r, nums):
        if l == r:
            self.s[o] = nums[l]
            return
        m = (l + r) // 2
        self._build(2 * o, l, m, nums)
        self._build(2 * o + 1, m + 1, r, nums)
        self.s[o] = self.s[2 * o] + self.s[2 * o + 1]

    def _apply(self, o, l, r):
        self.s[o] = (r - l + 1) - self.s[o]
        self.lz[o] = not self.lz[o]

    def _push(self, o, l, r):
        if self.lz[o]:
            m = (l + r) // 2
            self._apply(2 * o, l, m)
            self._apply(2 * o + 1, m + 1, r)
            self.lz[o] = False

    def _flip(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            self._apply(o, l, r)
            return
        self._push(o, l, r)
        m = (l + r) // 2
        if ql <= m:
            self._flip(2 * o, l, m, ql, qr)
        if qr > m:
            self._flip(2 * o + 1, m + 1, r, ql, qr)
        self.s[o] = self.s[2 * o] + self.s[2 * o + 1]

    def flip(self, ql, qr):
        self._flip(1, 0, self.n - 1, ql, qr)

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, o, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.s[o]
        self._push(o, l, r)
        m = (l + r) // 2
        res = 0
        if ql <= m:
            res += self._query(2 * o, l, m, ql, qr)
        if qr > m:
            res += self._query(2 * o + 1, m + 1, r, ql, qr)
        return res


def handle_queries(nums1, nums2, queries):
    st = SegTree(nums1)
    total = sum(nums2)
    res = []
    for t, p, _q in queries:
        if t == 1:
            st.flip(p, _q)
        elif t == 2:
            total += p * st.query(0, len(nums1) - 1)
        else:
            res.append(total)
    return res
```

C++：

```cpp
struct SegTree {
    int n;
    std::vector<long long> s;
    std::vector<bool> lz;
    explicit SegTree(const std::vector<int>& nums) : n(nums.size()), s(4 * n, 0), lz(4 * n, false) {
        build(1, 0, n - 1, nums);
    }
    void build(int o, int l, int r, const std::vector<int>& nums) {
        if (l == r) {
            s[o] = nums[l];
            return;
        }
        int m = (l + r) / 2;
        build(2 * o, l, m, nums);
        build(2 * o + 1, m + 1, r, nums);
        s[o] = s[2 * o] + s[2 * o + 1];
    }
    void applyNode(int o, int l, int r) {
        s[o] = (r - l + 1) - s[o];
        lz[o] = !lz[o];
    }
    void push(int o, int l, int r) {
        if (lz[o]) {
            int m = (l + r) / 2;
            applyNode(2 * o, l, m);
            applyNode(2 * o + 1, m + 1, r);
            lz[o] = false;
        }
    }
    void flip(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) {
            applyNode(o, l, r);
            return;
        }
        push(o, l, r);
        int m = (l + r) / 2;
        if (ql <= m) flip(2 * o, l, m, ql, qr);
        if (qr > m) flip(2 * o + 1, m + 1, r, ql, qr);
        s[o] = s[2 * o] + s[2 * o + 1];
    }
    long long query(int o, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return s[o];
        push(o, l, r);
        int m = (l + r) / 2;
        long long res = 0;
        if (ql <= m) res += query(2 * o, l, m, ql, qr);
        if (qr > m) res += query(2 * o + 1, m + 1, r, ql, qr);
        return res;
    }
    void flip(int ql, int qr) { flip(1, 0, n - 1, ql, qr); }
    long long query(int ql, int qr) { return query(1, 0, n - 1, ql, qr); }
};

std::vector<long long> handleQueries(const std::vector<int>& nums1, const std::vector<int>& nums2,
                                     const std::vector<std::array<long long, 3>>& queries) {
    SegTree st(nums1);
    long long total = 0;
    for (int v : nums2) total += v;
    std::vector<long long> res;
    for (auto& q : queries) {
        if (q[0] == 1) {
            st.flip(static_cast<int>(q[1]), static_cast<int>(q[2]));
        } else if (q[0] == 2) {
            total += q[1] * st.query(0, static_cast<int>(nums1.size()) - 1);
        } else {
            res.push_back(total);
        }
    }
    return res;
}
```

**复杂度**：每次查询 O(log n)；空间 O(n)。

**易错点**：

- `nums2` 的总和初值要算进去（`total = sum(nums2)`），别从 0 开始。
- 翻转标记是布尔量，`s = 长度 - s`，不要写成加法/减法。
- `total += p * sum(nums1)` 用的是**当前**的 `nums1`，所以要在处理类型 2 时实时查询。

**相似题**：区间取反的「离线差分」思路见第 4 篇；本题是「在线线段树」版本。

---

## 模式五：区间合并信息

**适用信号**：查询答案不能只靠一个数，而需要把左右两段的「边界信息」拼起来。

**核心动作**：为每个节点设计一套「足够合并」的状态。本题需要 5 个量：区间长度、左端字符及其
连续长度、右端字符及其连续长度、区间内最优值。父亲由两个孩子按固定规则合并。

### 2213. 由单个字符重复的最长子字符串（困难）

**题目**：每次把字符串 `s` 的某个下标改成新字符，问每次修改后「由单个字符重复形成的最长子串」
长度。

**思路**：

单点修改 + 全局最长同类连续段，用线段树。关键是每个节点存：

- `ln`：区间长度；
- `lc, ll`：最左端字符、从左端起连续相同字符的长度（左前缀）；
- `rc, rl`：最右端字符、从右端起连续相同字符的长度（右后缀）；
- `bs`：本区间内最长同类连续段。

合并左右 `L`、`R` 时：

- 左前缀：若 `L` 整段同色（`ll[L] == ln[L]`）且 `lc[L] == lc[R]`，则能延伸到 `R` 的左前缀，
  长度 `ln[L] + ll[R]`；否则就是 `ll[L]`。
- 右后缀：对称地，若 `R` 整段同色且 `rc[R] == rc[L]`，则为 `ln[R] + rl[L]`；否则 `rl[R]`。
- `bs`：先取 `max(bs[L], bs[R])`；若 `rc[L] == lc[R]`，还能把「`L` 的右后缀 + `R` 的左前缀」
  跨边界拼起来。

每次修改只更新一条到根的路径，根的 `bs` 就是答案。

**代码**：

Python：

```python
class SegTree:
    def __init__(self, s):
        self.n = len(s)
        self.s = s
        N = 4 * self.n
        self.ln = [0] * N
        self.lc = [0] * N
        self.ll = [0] * N
        self.rc = [0] * N
        self.rl = [0] * N
        self.bs = [0] * N
        self._build(1, 0, self.n - 1)

    def _build(self, o, l, r):
        if l == r:
            c = self.s[l]
            self.ln[o] = 1
            self.lc[o] = c
            self.ll[o] = 1
            self.rc[o] = c
            self.rl[o] = 1
            self.bs[o] = 1
            return
        m = (l + r) // 2
        self._build(2 * o, l, m)
        self._build(2 * o + 1, m + 1, r)
        self._pull(o)

    def _pull(self, o):
        left, right = 2 * o, 2 * o + 1
        self.ln[o] = self.ln[left] + self.ln[right]

        self.lc[o] = self.lc[left]
        self.ll[o] = self.ll[left]
        if self.ll[left] == self.ln[left] and self.lc[left] == self.lc[right]:
            self.ll[o] = self.ln[left] + self.ll[right]

        self.rc[o] = self.rc[right]
        self.rl[o] = self.rl[right]
        if self.rl[right] == self.ln[right] and self.rc[right] == self.rc[left]:
            self.rl[o] = self.ln[right] + self.rl[left]

        best = max(self.bs[left], self.bs[right])
        if self.rc[left] == self.lc[right]:
            best = max(best, self.rl[left] + self.ll[right])
        self.bs[o] = best

    def update(self, idx, ch):
        self._update(1, 0, self.n - 1, idx, ch)

    def _update(self, o, l, r, idx, ch):
        if l == r:
            self.lc[o] = ch
            self.rc[o] = ch
            return
        m = (l + r) // 2
        if idx <= m:
            self._update(2 * o, l, m, idx, ch)
        else:
            self._update(2 * o + 1, m + 1, r, idx, ch)
        self._pull(o)


def longest_repeating(s, query_characters, query_indices):
    st = SegTree(s)
    res = []
    for ch, i in zip(query_characters, query_indices):
        st.update(i, ch)
        res.append(st.bs[1])
    return res
```

C++：

```cpp
struct SegTree {
    int n;
    std::string s;
    std::vector<int> ln, lc, ll, rc, rl, bs;
    explicit SegTree(const std::string& str)
        : n(str.size()), s(str), ln(4 * n), lc(4 * n), ll(4 * n), rc(4 * n), rl(4 * n),
          bs(4 * n) {
        build(1, 0, n - 1);
    }
    void build(int o, int l, int r) {
        if (l == r) {
            ln[o] = 1;
            lc[o] = rc[o] = s[l];
            ll[o] = rl[o] = 1;
            bs[o] = 1;
            return;
        }
        int m = (l + r) / 2;
        build(2 * o, l, m);
        build(2 * o + 1, m + 1, r);
        pull(o);
    }
    void pull(int o) {
        int L = 2 * o, R = 2 * o + 1;
        ln[o] = ln[L] + ln[R];

        lc[o] = lc[L];
        ll[o] = ll[L];
        if (ll[L] == ln[L] && lc[L] == lc[R]) ll[o] = ln[L] + ll[R];

        rc[o] = rc[R];
        rl[o] = rl[R];
        if (rl[R] == ln[R] && rc[R] == rc[L]) rl[o] = ln[R] + rl[L];

        int best = std::max(bs[L], bs[R]);
        if (rc[L] == lc[R]) best = std::max(best, rl[L] + ll[R]);
        bs[o] = best;
    }
    void update(int o, int l, int r, int idx, char ch) {
        if (l == r) {
            lc[o] = rc[o] = ch;
            return;
        }
        int m = (l + r) / 2;
        if (idx <= m) update(2 * o, l, m, idx, ch);
        else update(2 * o + 1, m + 1, r, idx, ch);
        pull(o);
    }
    void update(int idx, char ch) { update(1, 0, n - 1, idx, ch); }
};

std::vector<int> longestRepeating(const std::string& s, const std::string& queryCharacters,
                                  const std::vector<int>& queryIndices) {
    SegTree st(s);
    std::vector<int> res;
    for (int i = 0; i < static_cast<int>(queryIndices.size()); ++i) {
        st.update(queryIndices[i], queryCharacters[i]);
        res.push_back(st.bs[1]);
    }
    return res;
}
```

**复杂度**：建树 O(n)，每次修改 O(log n)；空间 O(n)。

**易错点**：

- 「左前缀能延伸」的条件是「左孩子整段同色」**且**「左端字符 == 右孩子的左端字符」，两个条件
  缺一不可。
- 只有叶子修改时改字符，别忘了改完要 `_pull` 一路向上重算。
- 单点更新时叶子的 `ln/ll/rl/bs` 不变（长度恒为 1），只改字符即可。

**相似题**：区间最大子段和的线段树版是同一思路（节点维护和、前缀最大、后缀最大、整体最大），
见第 12 篇分治里的「跨越中点」；本题把「和」换成了「连续段长度」。

---

## 模式六：线段树上二分找第一个满足条件的位置

**适用信号**：要在某个区间里找「最靠左的、值大于阈值」的位置，且数组会变或规模大。

**核心动作**：线段树每个节点存区间最大值。查找时从根往下：先看左孩子最大值是否超阈值，超就
往左走，否则往右走——一次下降定位答案，O(log n)。

### 2940. 找到 Alice 和 Bob 可以相遇的建筑（困难）

**题目**：给定高度数组和查询 `[a, b]`。两人只能向右移动到**严格更高**的建筑上，求两人能共同
到达的最小编号建筑，不能则 -1。

**思路**：

先推导「能到达」的充要条件：从 `i` 出发能到 `j > i`，当且仅当 `heights[j] > heights[i]`。
必要性：严格递增链的终点必然比起点高；充分性：只要 `j` 更高，就能一步直接跳过去。于是：

- `a == b` → 答案就是 `a`；
- 令 `lo = min(a,b)`、`hi = max(a,b)`，答案不可能在 `hi` 左边。先看 `j = hi`：`hi` 那个人已就位，
  只需 `heights[hi] > heights[lo]`，成立则答案就是 `hi`；
- 否则在 `hi` 右侧找**最靠左**的高度严格大于 `max(heights[a], heights[b])` 的建筑。

「右侧第一个大于阈值的位置」用最大值线段树上二分：先看右侧区间最大值是否超阈值，不超直接 -1；
否则从根递归下降，优先左孩子，第一个「子树最大值 > 阈值」的叶子即答案。

**代码**：

Python：

```python
def leftmost_building_queries(heights, queries):
    n = len(heights)
    size = 1
    while size < n:
        size *= 2
    tree = [-1] * (2 * size)
    for i, h in enumerate(heights):
        tree[size + i] = h
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[2 * i], tree[2 * i + 1])

    def range_max(l, r):
        res = -1
        l, r = l + size, r + size + 1
        while l < r:
            if l & 1:
                res = max(res, tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, tree[r])
            l //= 2
            r //= 2
        return res

    def first_greater(l, t):
        if l >= n or range_max(l, n - 1) <= t:
            return -1

        def rec(o, nl, nr):
            if nr < l or tree[o] <= t:
                return -1
            if nl == nr:
                return nl
            mid = (nl + nr) // 2
            left = rec(2 * o, nl, mid)
            if left != -1:
                return left
            return rec(2 * o + 1, mid + 1, nr)

        return rec(1, 0, size - 1)

    res = []
    for a, b in queries:
        if a == b:
            res.append(a)
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if heights[hi] > heights[lo]:
            res.append(hi)
        else:
            res.append(first_greater(hi + 1, max(heights[a], heights[b])))
    return res
```

C++：

```cpp
std::vector<int> leftmostBuildingQueries(const std::vector<int>& heights,
                                         const std::vector<std::pair<int, int>>& queries) {
    int n = heights.size();
    int size = 1;
    while (size < n) size *= 2;
    std::vector<int> tree(2 * size, -1);
    for (int i = 0; i < n; ++i) tree[size + i] = heights[i];
    for (int i = size - 1; i > 0; --i) tree[i] = std::max(tree[2 * i], tree[2 * i + 1]);

    auto rangeMax = [&](int l, int r) {
        int res = -1;
        for (l += size, r += size + 1; l < r; l /= 2, r /= 2) {
            if (l & 1) res = std::max(res, tree[l++]);
            if (r & 1) res = std::max(res, tree[--r]);
        }
        return res;
    };
    auto firstGreater = [&](int l, int t) -> int {
        if (l >= n || rangeMax(l, n - 1) <= t) return -1;
        std::function<int(int, int, int)> rec = [&](int o, int nl, int nr) -> int {
            if (nr < l || tree[o] <= t) return -1;
            if (nl == nr) return nl;
            int mid = (nl + nr) / 2;
            int left = rec(2 * o, nl, mid);
            if (left != -1) return left;
            return rec(2 * o + 1, mid + 1, nr);
        };
        return rec(1, 0, size - 1);
    };

    std::vector<int> res;
    for (auto [a, b] : queries) {
        if (a == b) {
            res.push_back(a);
            continue;
        }
        int lo = std::min(a, b), hi = std::max(a, b);
        if (heights[hi] > heights[lo]) {
            res.push_back(hi);
        } else {
            res.push_back(firstGreater(hi + 1, std::max(heights[a], heights[b])));
        }
    }
    return res;
}
```

**复杂度**：建树 O(n)，每次查询 O(log n)；空间 O(n)。

**易错点**：

- 树开到「大于等于 n 的最小 2 的幂」`size`，多出来的叶子填 -1（比任何高度都小），不影响
  「找第一个更大」。
- 递归下降时先判 `nr < l`（区间在目标左边）再判 `tree[o] <= t`，两者都不能少。
- `heights[hi] > heights[lo]` 是「直接在 hi 相遇」的快速判断，别漏；否则会去 `hi+1` 找而错失
  `hi` 本身。

**相似题**：2407 的「值域最大值」；连续区间最值查询的稀疏表（ST 表）思路见第 12 篇分治的
区间查询延伸。

---

## 规律总结

1. **先问三件事**：合并算什么、查询怎么拆、有没有区间修改。这三个问题的答案基本决定了线段树
   长什么样。
2. **迭代 vs 递归**：只做单点修改 + 可合并查询，用自底向上的迭代线段树（307）最短最快；一旦
   出现区间修改（懒标记）或要在树上二分，用递归（其余各题）。
3. **懒标记的两种形态**：加法型（`lz += v`，下推累加，见日历三题/2569 的翻转）和赋值型
   （下推直接覆盖，见 715/699）。翻转可以看成对 0/1 的「异或赋值」，所以也归入后者。
4. **动态开点**：坐标大到 1e9 时用节点按需创建。**关键坑**：下推 `_push` 必须先无条件创建
   孩子，不能只在「有标记」时创建，否则递归会落到空节点。
5. **单位元**：合并的单位元要与运算匹配——求和是 0，求最大值是 -1 / -inf；查询完全不相交时返回
   单位元，这样合并才不会污染结果。
6. **节点状态要「够合并」**：2213 的五个量是最典型的例子；设计不出来时先问「左右拼起来还缺什么
   信息」。
7. **线段树上二分**：当查询形如「区间里第一个满足阈值的位置」，不需要单独二分再判断，直接在
   树上按「左孩子最大值是否达标」下降，一次 O(log n)。
8. **和树状数组的分工**：树状数组适合「单点改 + 前缀和/值域计数」，常数小；线段树适合「区间
   最值/可合并信息 + 区间修改」。能用树状数组就别上线段树。
9. **区间开闭要统一**：题目多半给半开区间 `[l, r)`，转成线段树的左闭右闭就是 `[l, r-1]`；一个
   端点的差别足以把「相邻」误判成「重叠」。

## 与其它篇的交叉

- 与第 24 篇树状数组：307 / 2407 / 值域计数类题都能用树状数组做；当需要「区间最值」或「区间
  修改」时，树状数组就顶不住，换线段树。493 翻转对（第 24 篇）也可以用线段树做，思路一致。
- 与第 4 篇前缀和：离线、可差分的区间修改（1109 航班预订、1094 拼车）用差分数组更简单；
  一旦要「在线」处理，才需要线段树（699）。
- 与第 12 篇分治：2213 的「合并左右信息」和分治里「跨越中点的答案」是同一个思考方式；2940 的
  区间最值查询也可用分治的 ST 表。
- 与第 13 篇动态规划：2407 是「用数据结构优化 DP 转移」的典型——把 `O(n)` 的转移用线段树压到
  `O(log n)`。
