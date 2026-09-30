# 前缀和与差分

前缀和解决的是**区间求和**问题。最朴素的做法是每问一次就把区间里的元素累加一遍，单次 O(n)；
但如果数组不会被修改、而查询有很多次，我们就可以先花 O(n) 建一张「累计快照」，
之后每次查询只用两个数相减、O(1) 出答案。这就是前缀和的全部思想——**把重复的累加提前算好**。

为什么两个数相减就能得到区间和？记 `prefix[i]` 为「前 i 个元素之和」。那么
`prefix[right+1]` 是从头加到 `right`、`prefix[left]` 是从头加到 `left-1`，两者相减，
共同的前半段 `[0, left-1]` 恰好抵消，剩下的正是 `[left, right]`。多留的 `prefix[0] = 0`
是为了让 `left = 0` 时也套用同一个公式，不必特判空区间。

前缀和的威力不止于「查询」：它把「子数组的元素和」变成了「两个前缀和的差」，于是很多
看似要枚举区间的问题，就被改写成了「找两个下标」的问题。加上一张哈希表记录前缀和出现的位置或次数，
就能处理**含负数**、滑动窗口无能为力的场景（560、974），这是本篇与 `hash` 分类交叉的地方。

「差分」是前缀和的逆运算：如果说前缀和擅长「多次查询区间和」，差分就擅长「多次修改区间、
最后一次性还原」。它和二维前缀和一起，构成本篇后半部分的两个新模式。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 一维前缀和 | 303. 区域和检索 - 数组不可变 | 简单 |
| 前缀和看平衡点 | 724. 寻找数组的中心下标 | 简单 |
| 二维前缀和 | 304. 二维区域和检索 - 矩阵不可变 | 中等 |
| 差分 | 1109. 航班预订统计 | 中等 |
| 差分 | 1094. 拼车 | 中等 |
| 前缀和 + 哈希（交叉） | 560. 和为 K 的子数组 | 中等 |

---

## 模式一：一维前缀和（预存快照）

**适用信号**：数组**不会被修改**，但要**反复**查询若干区间的元素和。关键词是「不可变」+「多次查询」。

核心动作：用一次遍历建出 `prefix`，`prefix[i+1] = prefix[i] + nums[i]`，
查询时套公式 `prefix[right+1] - prefix[left]`。建表是一次性的，之后每次查询都是 O(1)。

### 303. 区域和检索 - 数组不可变（简单）

**题目**：设计一个类 `NumArray`，用整数数组 `nums` 初始化，支持查询下标区间 `[left, right]` 内所有元素之和。数组在整个过程中不会被修改。例如 `nums = [-2, 0, 3, -5, 2, -1]`，`sum_range(0, 2) = 1`，`sum_range(2, 5) = -1`。

**思路（预存前缀和，查询相减）**：
每次查询都累加一遍太慢，我们把「从头到某处」的和提前存好。建一个长度为 `n + 1` 的数组
`prefix`，`prefix[0] = 0`，`prefix[i+1] = prefix[i] + nums[i]`，也就是 `prefix[i]` 表示
`nums` 前 `i` 个元素之和。

查询 `[left, right]` 时返回 `prefix[right + 1] - prefix[left]`：

- `prefix[right + 1]` 覆盖 `nums[0..right]`；
- `prefix[left]` 覆盖 `nums[0..left-1]`；
- 相减后，`[0, left-1]` 被减掉，只剩 `nums[left..right]`。

为什么 `prefix` 要比 `nums` 多一位：这一位是 `prefix[0] = 0`，代表「空前缀」。
有了它，`left = 0` 的查询也能统一写成 `prefix[right+1] - prefix[0] = prefix[right+1]`，
不用为「从数组开头开始」这种边界单独写一个 `if`。这种「多留一位零」是前缀和的标准写法。

**代码**（完整可运行版见 `src/prefix-sum/range_sum_query.py` / `.cpp`）：

```python
class NumArray:
    def __init__(self, nums):
        self.prefix = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            self.prefix[i + 1] = self.prefix[i] + x

    def sum_range(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

```cpp
class NumArray {
public:
    explicit NumArray(const std::vector<int> &nums) {
        prefix_.resize(nums.size() + 1, 0);
        for (std::size_t i = 0; i < nums.size(); ++i)
            prefix_[i + 1] = prefix_[i] + nums[i];
    }

    int sumRange(int left, int right) const {
        return prefix_[right + 1] - prefix_[left];
    }

private:
    std::vector<int> prefix_;
};
```

- **复杂度**：预处理时间 O(n)、空间 O(n)；每次查询时间 O(1)。
- **易错点**：`prefix` 长度是 `n + 1`，别写成 `n`，否则 `prefix[left]` 会越界；下标换算容易差一位，记住 `prefix[i]` 对应「前 i 个元素」、对应原数组 `[0, i-1]`；数组元素可能为负，所以「区间和」可正可负，但公式一视同仁，不用特判。
- **相似题**：304. 二维区域和检索（把一维前缀和推广到矩阵，用容斥原理，见本分类续篇）；560. 和为 K 的子数组（把「查询区间和」反过来用成「找两个前缀和之差」，见下）；238. 除自身以外数组的乘积（前缀积思想，见本分类续篇）。

---

## 模式二：前缀和看「平衡点」

**适用信号**：要找一个位置，使它**左边**和**右边**的某种累积量满足相等（或某种关系）。
本质是「用整段和减去左边部分，得到右边部分」，避免对每个位置分别重算两侧。

核心动作：先求整段总和 `total`，扫描时用一个变量 `left` 维护「当前元素左侧之和」。
在位置 `i`，右侧之和就等于 `total - left - nums[i]`，直接比较即可。

### 724. 寻找数组的中心下标（简单）

**题目**：给定整数数组 `nums`，找出「中心下标」——该下标左侧所有元素之和等于右侧所有元素之和。若存在多个，返回最左边那个；不存在返回 `-1`。规定下标 0 左侧为空、和为 0。例如 `nums = [1, 7, 3, 6, 5, 6]`，中心下标是 3（左侧 `1+7+3=11`，右侧 `5+6=11`）。

**思路（整段和减去左右两侧）**：
对每个下标分别求左右两边之和是 O(n²)。换个角度：整段 `total` 是固定的，
只要用一个变量 `left` 边走边累加「当前元素左边的和」，那么

> 当前元素右侧之和 = `total − left − nums[i]`

于是判断 `left == total - left - nums[i]` 即可。三个量都是现成的，每个位置 O(1)。

为什么从左往右扫一次就够：题目要的是**最左边**的中心下标，第一个满足条件的位置就是答案，
找到即可返回，不需要继续往右找（往右只会更靠右）。

为什么「先判断、后累加」：`left` 必须表示「**不含**当前元素」的左侧和。
如果先把 `nums[i]` 加进 `left`，再拿去判断，就等于把当前元素同时算进了左边，
等式两边都会错。顺序写反是本题最常见的 bug。

**代码**（`src/prefix-sum/find_pivot_index.py` / `.cpp`）：

```python
def pivot_index(nums):
    total = sum(nums)
    left = 0
    for i, x in enumerate(nums):
        if left == total - left - x:
            return i
        left += x
    return -1
```

```cpp
int pivotIndex(const std::vector<int> &nums) {
    long long total = std::accumulate(nums.begin(), nums.end(), 0LL);
    long long left = 0;
    for (std::size_t i = 0; i < nums.size(); ++i) {
        if (left == total - left - nums[i]) return static_cast<int>(i);
        left += nums[i];
    }
    return -1;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：必须先判断、后累加 `left`（见上）；数组首元素就是中心下标时左侧为 0，公式自动满足，验证 `[1]` 应得 0；全零数组 `[0, 0, 0]` 的最左中心下标是 0，不要误返回中间；元素可正可负，所以「和相等」不能用单调性加速，只能老老实实扫一遍；C++ 里用 `long long` 存和，避免求和溢出。
- **相似题**：1991. 找到数组的中间位置（同题不同叫法）；1422. 分割字符串的最大得分（同样把「分割点」两侧的统计量拿来比较）；560. 和为 K 的子数组（也是「两侧前缀量做差」，但目标是找配对而非找平衡点，见下）。

---

## 模式三：二维前缀和（矩阵上的容斥）

**适用信号**：矩阵**不会被修改**，但要**反复**查询某个子矩阵（矩形区域）的元素和。
与一维的唯一区别是：矩形有上下左右四条边，要把「两块相减」升级成「四块容斥」。

核心动作：`prefix[i][j]` 表示以 `(0,0)` 为左上角、`(i-1, j-1)` 为右下角的子矩阵和，
比原矩阵多一行一列（第 0 行/列全为 0）。建表与查询都套同一个容斥式。

### 304. 二维区域和检索 - 矩阵不可变（中等）

**题目**：给定二维矩阵 `matrix`，支持多次查询子矩阵的元素和。子矩阵由左上角 `(row1, col1)` 和右下角 `(row2, col2)` 确定，矩阵不会被修改。例如 `matrix = [[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]`，`sum_region(2,1,4,3) = 8`，`sum_region(1,1,2,2) = 11`。

**思路（二维前缀和 + 容斥）**：
一维前缀和里，区间和 = 两个前缀相减。矩阵里的「区间」是一个矩形，它的边界不止一个方向，
所以要相减的也不止两个。设 `prefix[i][j]` 为「以 `(0,0)` 为左上角、`(i-1, j-1)` 为右下角」的
子矩阵和。建表时，`prefix[i+1][j+1]` 对应的矩形可以拆成：

> 当前格子 `matrix[i][j]` + 上方的矩形 `prefix[i][j+1]` + 左方的矩形 `prefix[i+1][j]`
> − 左上角被重复加了一次的矩形 `prefix[i][j]`

写成公式：

```
prefix[i+1][j+1] = matrix[i][j] + prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
```

这就是**容斥原理**：先把两块相邻区域加起来，再减去它们重叠的部分。

查询的道理完全相同。要矩形 `(row1, col1)` 到 `(row2, col2)` 的和，先取「到右下角的大矩形」，
减去它上面的一条、左边的一条，再加回被减了两次的左上角小矩形：

```
prefix[row2+1][col2+1] - prefix[row1][col2+1] - prefix[row2+1][col1] + prefix[row1][col1]
```

其中多的第 0 行、第 0 列是为了让 `row1 = 0` 或 `col1 = 0` 时公式仍然成立，不必特判越界，
和一维里 `prefix[0] = 0` 的作用一模一样。

**为什么不用「每行各做一维前缀和、查询时逐行相加」**：那样每次查询仍要扫过 O(m) 行，
二维前缀和把这步也省掉，查询直接 O(1)；代价不过是预处理从 O(m·n) 变成同一量级的二维建表。

**代码**（完整可运行版见 `src/prefix-sum/range_sum_query_2d.py` / `.cpp`）：

```python
class NumMatrix:
    def __init__(self, matrix):
        m = len(matrix)
        n = len(matrix[0]) if m else 0
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (
                    matrix[i][j]
                    + self.prefix[i][j + 1]
                    + self.prefix[i + 1][j]
                    - self.prefix[i][j]
                )

    def sum_region(self, row1, col1, row2, col2):
        return (
            self.prefix[row2 + 1][col2 + 1]
            - self.prefix[row1][col2 + 1]
            - self.prefix[row2 + 1][col1]
            + self.prefix[row1][col1]
        )
```

```cpp
class NumMatrix {
public:
    explicit NumMatrix(const std::vector<std::vector<int>> &matrix) {
        int m = static_cast<int>(matrix.size());
        int n = m ? static_cast<int>(matrix[0].size()) : 0;
        prefix_.assign(m + 1, std::vector<long long>(n + 1, 0));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                prefix_[i + 1][j + 1] = matrix[i][j] + prefix_[i][j + 1] +
                                        prefix_[i + 1][j] - prefix_[i][j];
    }

    int sumRegion(int row1, int col1, int row2, int col2) const {
        return static_cast<int>(prefix_[row2 + 1][col2 + 1] -
                                prefix_[row1][col2 + 1] -
                                prefix_[row2 + 1][col1] + prefix_[row1][col1]);
    }

private:
    std::vector<std::vector<long long>> prefix_;
};
```

- **复杂度**：预处理时间 O(m·n)、空间 O(m·n)；每次查询 O(1)。
- **易错点**：`prefix` 是 `(m+1) × (n+1)`，别和 `matrix` 同尺寸，否则 `prefix[i+1]` 越界；容斥式的四个项符号是 `+ − − +`，别漏掉最后那个「加回重叠」的项，否则角落会被多减；传参顺序是 `(row1, col1, row2, col2)` 而不是两个点坐标；C++ 用 `long long` 存前缀和，避免大量元素求和溢出。
- **相似题**：1292. 元素和小于等于阈值的正方形的最大边长（在二维前缀和上二分边长）；1074. 元素和为目标值的子矩阵数量（固定上下边界后，把每列的和压缩成一维，再用 560 的「前缀和 + 哈希」统计）；221. 最大正方形（虽然用 DP，但判断某块是否全 1 也常用二维前缀和辅助）。

---

## 模式四：差分（多次区间修改，最后一次性还原）

**适用信号**：需要对一段区间**整体加上（或减去）同一个数**，这种操作要做很多次，
最后才问每个位置最终的值。关键词是「区间修改」——它是前缀和的镜像。

核心动作：维护一个差分数组 `diff`，对区间 `[l, r]` 加 `v` 只改两个点：

```
diff[l] += v
diff[r + 1] -= v
```

为什么这样有效：把 `diff` 做前缀和时，`l` 处的 `+v` 会从此一直传递下去，
而 `r+1` 处的 `-v` 恰好在区间结束后把这份增量抵消，于是前缀和只在 `[l, r]` 内多了 `v`。
一次区间修改因此从 O(区间长度) 降到 O(1)，代价是最后要 O(n) 求一次前缀和还原。
**前缀和与差分是一对逆运算**：前缀和把「点值」变成「累计量」，差分把「区间修改」变成「点修改」。

### 1109. 航班预订统计（中等）

**题目**：有 `n` 个航班，编号 `1..n`。给定预订记录 `bookings`，每条是 `[first, last, seats]`，表示从 `first` 到 `last`（含两端）每个航班都被预订了 `seats` 个座位。返回长度为 `n` 的数组，第 `i` 项是第 `i+1` 个航班的总预订数。例如 `n = 5`，`bookings = [[1,2,10],[2,3,20],[2,5,25]]`，结果为 `[10, 55, 45, 25, 25]`。

**思路（区间加 → 差分两个单点改）**：
朴素做法是对每条预订，把 `[first, last]` 里的航班逐个累加，最坏 O(n·m)。
这正是「多次区间加、最后统一问结果」的典型场景，用差分：

- `diff[first - 1] += seats`（输入是 1 下标，转成 0 下标）；
- `diff[last] -= seats`（`last` 航班对应 0 下标 `last-1`，它的下一位就是 `last`）。

把所有预订处理完，对 `diff` 从头做一次前缀和，每个位置累计到的值就是该航班的总预订数。
为什么减在 `last` 而不是 `last + 1`：因为输入区间两端是**闭**的 `[first, last]`，
`last` 之后才失效；用 0 下标表示时「after last」正好是下标 `last`。

**代码**（`src/prefix-sum/corporate_flight_bookings.py` / `.cpp`）：

```python
def corp_flight_bookings(bookings, n):
    diff = [0] * (n + 1)
    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats
    res = [0] * n
    cur = 0
    for i in range(n):
        cur += diff[i]
        res[i] = cur
    return res
```

```cpp
std::vector<long long> corpFlightBookings(const std::vector<std::vector<int>> &bookings,
                                          int n) {
    std::vector<long long> diff(n + 1, 0);
    for (const auto &b : bookings) {
        int first = b[0], last = b[1], seats = b[2];
        diff[first - 1] += seats;
        diff[last] -= seats;
    }
    std::vector<long long> res(n, 0);
    long long cur = 0;
    for (int i = 0; i < n; ++i) {
        cur += diff[i];
        res[i] = cur;
    }
    return res;
}
```

- **复杂度**：时间 O(n + m)（`m` 为预订条数），空间 O(n)。
- **易错点**：`diff` 要开 `n + 1`，因为 `diff[last]` 中 `last` 可以等于 `n`（覆盖到最后一个航班时），开 `n` 会越界；1 下标转 0 下标时是 `first-1` 和 `last`（减一后再取「下一位」），差一位是常见错误；求和结果可能超过 `int`，C++ 用 `long long`。
- **相似题**：1094. 拼车（同款差分，只是把「座位数」换成「车上人数」并加一个容量上限，见下）；370. 区间加法（差分模板题）；2381. 字母移位 II（把字母的循环移位看成区间加，用差分并处理取模）。

### 1094. 拼车（中等）

**题目**：一辆车最多坐 `capacity` 人，只朝一个方向开。给定行程 `trips`，每条是 `[numPassengers, from, to]`，表示在 `from` 站上 `numPassengers` 人、在 `to` 站下车。判断能否把所有乘客运达（任意时刻车上人数不超过容量）。例如 `trips = [[2,1,5],[3,3,7]]`，`capacity = 4`，返回 `False`。

**思路（站点差分 + 容量检查）**：
每个行程相当于给「从 `from` 到 `to` 之间的站点区间」整体加上 `numPassengers`，
又是区间加——用差分：`diff[from] += num`、`diff[to] -= num`。
然后从头累加 `diff`，边走边检查车上人数是否超过 `capacity`，一旦超过立刻返回 `False`。

为什么减在 `to` 而不是 `to + 1`：`to` 是下车站，到达 `to` 时乘客已经离开，
所以这个行程占用的区间是**左闭右开** `[from, to)`。这和 1109 不同，
1109 是闭区间 `[first, last]`，所以那里减在下标 `last`（即 `last+1` 的 0 下标写法），
本题直接减在下标 `to`。**判断区间开闭，是差分题最容易出错的地方**。

**代码**（`src/prefix-sum/car_pooling.py` / `.cpp`）：

```python
def car_pooling(trips, capacity):
    size = 0
    for _, _, end in trips:
        size = max(size, end)
    diff = [0] * (size + 1)
    for num, start, end in trips:
        diff[start] += num
        diff[end] -= num
    cur = 0
    for i in range(size):
        cur += diff[i]
        if cur > capacity:
            return False
    return True
```

```cpp
bool carPooling(const std::vector<std::vector<int>> &trips, int capacity) {
    int size = 0;
    for (const auto &t : trips)
        if (t[2] > size) size = t[2];
    std::vector<long long> diff(size + 1, 0);
    for (const auto &t : trips) {
        int num = t[0], start = t[1], end = t[2];
        diff[start] += num;
        diff[end] -= num;
    }
    long long cur = 0;
    for (int i = 0; i < size; ++i) {
        cur += diff[i];
        if (cur > capacity) return false;
    }
    return true;
}
```

- **复杂度**：时间 O(S + m)（`S` 为最大站点编号，`m` 为行程数），空间 O(S)。
- **易错点**：区间是左闭右开 `[from, to)`，`diff[to] -= num`，不是 `to+1`；检查要在**上车之后**进行（先 `cur += diff[i]` 再比较），否则会漏掉「刚上车就超载」的情况；`diff` 长度按最大站点编号申请，遍历时只到 `size`（不含），保证最后一个下车站不影响检查；`cur` 用 `long long`。
- **相似题**：1109. 航班预订统计（同款差分，区别只在闭区间与开区间）；253. 会议室 II（把会议起止时间看成上下车，求同时进行的最大会议数，本质上就是差分前缀和的峰值）；732. 我的日程安排表 III（动态区间加的差分进阶）。

---

## 交叉：560. 和为 K 的子数组（前缀和 + 哈希）

560 已收录在 `02-hash`，这里从**前缀和**的视角再看一遍，因为它正是「前缀和 + 哈希」这一组合的起点，
也是本分类续篇 974 的直接前身。

回忆 303：子数组 `nums[j..i-1]` 的和等于 `prefix[i] - prefix[j]`。要让它恰好等于 `k`，
等价于 `prefix[j] == prefix[i] - k`。于是问题变成：从左往右扫描，对每个 `prefix[i]`，
数一数「前面出现过多少个前缀和等于 `prefix[i] - k`」。把前缀和当作键、出现次数当作值存进哈希表，
再用 `count[0] = 1` 预置空前缀，就能一次线性扫描统计完。

为什么这里必须用哈希、不能沿用滑动窗口：滑动窗口依赖「窗口越长和越大」的单调性，
可数组一旦含负数，前缀和就不再单调，收缩方向无从判断。哈希表不依赖顺序，只记录「谁出现过、几次」，
所以能处理任意整数。**这是前缀和类题目最重要的一次视角转换**：从「维护一段」变成「找两个前缀」。

完整代码与逐行讲解见 `02-hash` 的《和为 K 的子数组》一节。

- **相似题**：974. 和可被 K 整除的子数组（把「差 = k」换成「差 ≡ 0 (mod K)」，用「同余前缀和」配对，见本分类续篇）；523. 连续的子数组和（同余前缀和的另一变体，且要求长度至少为 2）；1248. 统计「优美子数组」（把奇偶性映射成 0/1 后用完全相同的前缀和计数）。

---

## 规律总结

1. **前缀和的三要素：多留一位零、公式相减、一次性预处理**。一维 `prefix` 长度为 `n + 1` 且 `prefix[0] = 0`，查询 `[left, right]` 用 `prefix[right+1] - prefix[left]`。二维则把数组扩成 `(m+1) × (n+1)`，用容斥式的 `+ − − +` 四块拼出任意矩形。两者都靠多留的那圈零兜住从 0 开始的边界。
2. **先问「数组变不变」**。不可变 + 多查询 → 前缀和正合适；如果数组会被**修改**、又要频繁查询区间和，普通前缀和就失效了，得换树状数组/线段树（本专题暂不展开）。
3. **前缀和的本质是「把区间问题变成两个端点的差」**。抓住这句话，很多题会自动简化：303 是「查两个前缀的差」，560 是「找两个前缀的差等于 k」，724 是「找使左右两侧差为零的分割点」，304 是「四个前缀的容斥」。
4. **含负数时前缀和不单调，别硬套滑动窗口**。560 / 974 这类题一旦有负数，窗口无法定向收缩，必须改用「前缀和 + 哈希」。
5. **差分是前缀和的逆运算**。前缀和面向「多次查询」，差分面向「多次修改」：区间 `[l, r]` 整体加 `v`，就做 `diff[l] += v`、`diff[r+1] -= v`，最后求一次前缀和还原。它的难点不在公式，而在**判断区间是闭还是开**——闭区间减在 `r+1`，开区间减在 `r`（见 1109 与 1094 的对比）。
6. **以 `src` 为准，docs 只做粘贴**。题解里的代码必须和 `code/leetcode/src/prefix-sum/` 下的实现**逐字一致**，避免文档与可运行代码脱节。
7. **相似题就是换一个「相等条件」或「区间语义」**。560 问「差 = k」，974 问「差能被 K 整除」，523 问「差能被 K 整除且下标间隔够远」——框架完全相同，只是配对条件不同；1109 与 1094 框架也完全相同，只是区间开闭不同。先把一个吃透，再看同类就是顺水推舟。
