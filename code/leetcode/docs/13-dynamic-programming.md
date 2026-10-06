# 动态规划（一）：线性、序列、背包、网格、区间、状态机与通用框架

动态规划（dynamic programming, DP）听起来吓人，其实做的事很朴素：**把一个问题拆成一串互相
重叠的小问题，先算出小问题的答案存起来，再用它们拼出大问题的答案。** 它和分治最大的区别
在于——分治的子问题彼此独立、算完就忘；DP 的子问题**互相重叠**，所以必须把结果记下来复用，
否则会一遍遍重复计算。

写 DP 时，只要老实回答四个问题，代码基本就出来了：

1. **状态定义**：`dp[i]`（或 `dp[i][j]`）到底表示什么？这是最关键的一步。
2. **转移方程**：`dp[i]` 能从哪些更小的状态推出来？
3. **初始化**：最小的、无法再分的状态值是多少？
4. **遍历顺序**：按什么顺序填表，才能保证用到的状态都已经算好？

本专题从最干净的一维线性 DP 起步。你会看到同一个骨架——「用一个或几个前驱状态算出当前
状态」——如何长出「数方案」「求花费」「选或不选」「处理环形」四种不同的脸。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：一维递推（数方案用加法） | 509. 斐波那契数 · 70. 爬楼梯 | 简单 |
| 模式二：结果本身就是一张表 | 118. 杨辉三角 | 简单 |
| 模式三：带权递推（取 min / max） | 746. 使用最小花费爬楼梯 · 198. 打家劫舍 | 简单 / 中等 |
| 模式四：把「环」剪成「线」 | 213. 打家劫舍 II | 中等 |
| 模式五：以「我」结尾的连续段 | 53. 最大子数组和 · 674. 最长连续递增序列 · 718. 最长重复子数组 | 中等 |
| 模式六：不要求连续的子序列 | 300. 最长递增子序列 · 1143. 最长公共子序列 | 中等 |
| 模式七：回文类区间问题 | 5. 最长回文子串 | 中等 |
| 模式八：两个前缀之间的距离 | 72. 编辑距离 | 困难 |
| 模式九：完全背包（物品可重复取） | 322. 零钱兑换 · 518. 零钱兑换 II · 279. 完全平方数 | 中等 |
| 模式十：0/1 背包（每件至多取一次） | 416. 分割等和子集 · 494. 目标和 · 1049. 最后一块石头的重量 II | 中等 |
| 模式十一：把状态切成「前缀 + 尾段」 | 139. 单词拆分 | 中等 |
| 模式十二：网格上的二维 DP | 62. 不同路径 · 63. 不同路径 II · 64. 最小路径和 · 221. 最大正方形 | 中等 |
| 模式十三：区间 DP（枚举「最后一步」） | 312. 戳气球 · 516. 最长回文子序列 · 647. 回文子串 | 中等 / 困难 |
| 模式十四：状态机 DP（股票买卖） | 121 · 122 · 123 · 188 · 309 · 714 买卖股票系列 | 简单 ~ 困难 |
| 综合：DP 的通用框架（五步流水线） | 152. 乘积最大子数组 · 32. 最长有效括号 | 中等 / 困难 |

同一模式下的题目放在一起，先读第一道、再体会第二道只多了哪一点。**学 DP 的重点不是背题，
而是练熟「定义状态 → 写转移 → 定初值 → 排顺序」这条流水线。**

前半部分（模式一~四）是一维线性 DP，中间部分（模式五~八）进入**序列与双序列 DP**：状态
从「一个下标」变成「一个下标」或「两个下标」。再往后，模式九~十一是**背包与切割 DP**：
状态还是「容量」，但把「一件物品取几次」「前缀能不能被切开」这些新语义安进转移里，又会
看到「正序还是倒序」「物品在外还是容量在外」这类只差一行的新讲究。最后模式十二、十三把
状态从「一条线」推向**二维网格与闭区间**：转移开始在「上方 / 左方」和「区间内部的一点」
上做文章。最后的模式十四换一种视角——状态不再是「某个下标」，而是**手的姿态**（持股 / 空仓
/ 冷冻），一天一天地对这些姿态做切换，这就是「状态机 DP」。

走完这十四种模式，篇末再用两道综合题，把分散的招式收束成一条**写 DP 的通用流水线**：
无论题目长什么样，都从「状态定义」出发，一路走到「举例验证」。把这五步当成肌肉记忆，
新题无非是在某个环节上多绕一道弯。

---

## 模式一：一维递推（数方案用加法）

**适用信号**：每个位置只能由它前面有限个位置到达，且「到达某位置的方案数」可以累加。

**核心动作**：定义 `dp[i]` 为「到达/得到第 i 个状态的方案数」，转移就是把所有能一步走到
`i` 的前驱状态的方案数加起来；因为只依赖前两项，可以用滚动变量省掉整张表。

### 509. 斐波那契数（简单）

**题目**：斐波那契数 `F(0) = 0`，`F(1) = 1`，之后 `F(n) = F(n-1) + F(n-2)`。给定 `n`，计算 `F(n)`。

**思路**：

直接按定义写递归 `f(n) = f(n-1) + f(n-2)` 是对的，但会**指数级重复计算**：算 `f(5)` 时
要算 `f(4)` 和 `f(3)`，算 `f(4)` 时又要再算一遍 `f(3)`……同一批子问题被反复求解。

DP 的做法是换一种问法：与其「从上往下递归」，不如「从下往上填表」。因为每个数只由前两个数
决定，我们先放好 `F(0) = 0`、`F(1) = 1`，然后一路推到 `F(n)`。每个状态只算一次，快的多。

**为什么这就是 DP**：它包含了 DP 的全部四要素——状态 `dp[i] = F(i)`；转移
`dp[i] = dp[i-1] + dp[i-2]`；初始化 `dp[0]=0, dp[1]=1`；遍历顺序 i 从小到大。斐波那契是
DP 的最小样本，把它看透，后面只是状态变复杂而已。

**为什么可以用两个变量代替整张表**：当前值只用到前两项，更早的值再也用不上了。于是用
`prev2`、`prev1` 两个滚动变量边推边覆盖，空间从 O(n) 降到 O(1)。

**代码**（完整可运行版见 `src/dynamic-programming/fibonacci.py` / `.cpp`）：

```python
def fib(n):
    if n < 2:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```

```cpp
int fib(int n) {
    if (n < 2) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; ++i) {
        int cur = prev2 + prev1;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}
```

- **复杂度**：时间 O(n)（每个状态算一次），空间 O(1)（两个滚动变量）。
- **易错点**：`n < 2` 要单独返回 `n`（`F(0)=0`、`F(1)=1`），漏掉会下溢；滚动更新在 Python
  里靠**元组同时赋值**（右边先整体求值），若写成两条独立赋值会把 `prev1` 提前覆盖；C++ 则
  必须用临时变量 `cur` 中转，两条赋值顺序写反就错。
- **相似题**：70. 爬楼梯（递推式完全相同，初始项不同，见下）；509 的递归 + 记忆化解
  （把「自上而下 + 缓存」与「自下而上」对照理解）；746、198（把「加法」换成 `min`/`max`
  的带权版本，见模式三）。

### 70. 爬楼梯（简单）

**题目**：需要 `n` 阶才能到达楼顶，每次可以爬 1 或 2 个台阶，有多少种不同的方法可以爬到楼顶？

**思路**：

设 `dp[i]` 表示爬到第 `i` 阶的方法数。最后一步只有两种可能：从第 `i-1` 阶迈 1 步，或从
第 `i-2` 阶迈 2 步。这两类走法**互不重叠**（最后一步不同），所以把方案数相加：

```
dp[i] = dp[i-1] + dp[i-2]
```

初始化 `dp[0] = 1`（站在地面，算一种「还没走」的状态）、`dp[1] = 1`（只能迈 1 步到第 1 阶）。

**为什么「相加」而不是取最值**：题目问的是**有多少种方法**，不同到达方式都要计入总数，所以
用加法计数。后面 746 问的是**最小花费**，才改用 `min`。同一套状态、同一个前驱集合，聚合
运算符决定了题目问的是什么。

**为什么和斐波那契是同一道题**：递推式一模一样，只是起点从 `(0,1)` 变成 `(1,1)`。这正是
DP 的迁移性——状态和转移一旦对上，题目换了皮也认得出来。

**代码**（`src/dynamic-programming/climbing_stairs.py` / `.cpp`）：

```python
def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```

```cpp
int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; ++i) {
        int cur = prev2 + prev1;
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：初值是 `(1, 2)` 不是 `(0, 1)`，`dp[0]=1` 的定义（地面算一种空走法）要想清楚，
  不然 `n=1,2` 会偏差；`n <= 2` 直接返回 `n`，可与循环衔接一致；同样注意滚动赋值的顺序。
- **相似题**：509. 斐波那契数（同一递推式，见上）；746. 使用最小花费爬楼梯（把「数方案」
  改成「求最小花费」）；198. 打家劫舍（相邻约束下的最优选择，见模式三）。

---

## 模式二：结果本身就是一张表

**适用信号**：题目要输出的不是一个数，而是一整张二维结构；每个格子由它上方/左上的格子推出。

**核心动作**：直接开出与答案同形状的 `dp` 表，按「上行 → 下行」逐格填；边界格按题意赋初值。

### 118. 杨辉三角（简单）

**题目**：给定非负整数 `numRows`，生成杨辉三角的前 `numRows` 行。每行首尾是 1，中间每个数
等于它**上方**和**左上方**两个数之和。

**思路**：

状态是二维的：`dp[i][j]` 表示第 `i` 行第 `j` 个元素（行列都从 0 开始）。转移方程直接从图形
读出来——除首尾的 1 之外：

```
dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
```

首尾固定为 1。实现上可以先把整行填成 1，再只更新中间下标 `j = 1 .. i-1`，首尾自然保持 1，
省去特判。

**为什么遍历顺序是「上行→下行」**：算第 `i` 行时要用第 `i-1` 行的两个邻居，所以必须先算完
上一行。同一行内 `j` 从左到右或从右到左都不影响（本行格子之间不互相依赖），逐行生成天然满足。

**为什么不需要压缩空间**：和前面几题不同，这里的 `dp` 表**就是题目要求的输出**，不是中间
缓存。所以老老实实构造每一行、存进结果即可，压缩反而多此一举。

**代码**（`src/dynamic-programming/pascals_triangle.py` / `.cpp`）：

```python
def generate(num_rows):
    triangle = []
    for i in range(num_rows):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)
    return triangle
```

```cpp
std::vector<std::vector<int>> generate(int numRows) {
    std::vector<std::vector<int>> triangle;
    for (int i = 0; i < numRows; ++i) {
        std::vector<int> row(i + 1, 1);
        for (int j = 1; j < i; ++j) {
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
        }
        triangle.push_back(row);
    }
    return triangle;
}
```

- **复杂度**：时间 O(numRows²)（每行长度线性递增，总格数约 numRows²/2），空间 O(numRows²)
  （输出本身，无法省略）。
- **易错点**：中间循环是 `j = 1 .. i-1`（左闭右开，`range(1, i)` / `j < i`），写成 `j <= i`
  会把末尾的 1 覆盖成越界访问；第 0 行只有一个 1，此时内层循环一次都不执行，要保证
  `range(1, 0)` / `1 < 0` 是空的；构造 `row` 时先填满 1 是关键一步。
- **相似题**：119. 杨辉三角 II（只要第 `k` 行，可以只用一个数组原地滚动）；不同路径类
  （62. 不同路径、64. 最小路径和）也是「二维网格逐格递推」，只是转移方程取自网格移动方向。

---

## 模式三：带权递推（取 min / max）

**适用信号**：每一步都有一个代价或收益，要求「总代价最小」或「总收益最大」，且当前状态只
由前驱状态加/减一个值得到。

**核心动作**：转移时先用 `min` 或 `max` 在前驱状态里选出「最优的那个前驱」，再叠加本步的
代价/收益；答案落在最后一步的若干候选里。

### 746. 使用最小花费爬楼梯（简单）

**题目**：`cost[i]` 是从第 `i` 个台阶向上爬需要支付的费用，付完可以选择爬 1 或 2 个台阶。
可以从下标 0 或 1 开始爬。返回到达楼顶（最后一个台阶之上）的最低花费。

**思路**：

设 `dp[i]` 表示「到达第 `i` 个台阶并支付了它」的最小累计花费。上一步只能来自第 `i-1` 或
第 `i-2` 个台阶，取更便宜的那条路：

```
dp[i] = min(dp[i-1], dp[i-2]) + cost[i]
```

初始化 `dp[0] = cost[0]`、`dp[1] = cost[1]`（可以任选起点）。最后站到顶时，可以从最后一阶
（`n-1`）迈一步、也可以从倒数第二阶（`n-2`）迈两步，所以答案是 `min(dp[n-1], dp[n-2])`。

**为什么和爬楼梯（70）是「同一条骨架」**：状态和前驱完全相同，只是 70 用 `+` 数方案、
本题用 `min` 求最小花费。把这两题并排看，就能体会到「转移方程里的聚合运算符 = 题目在
问什么」。

**为什么初始状态与答案位置都要多想一层**：起点可以是 0 或 1，所以初值同时给了两个；终点在
「最后一阶之上」，所以不能直接回 `dp[n-1]`，要再看一眼前一步能不能从 `n-2` 直接跨过去。

**代码**（`src/dynamic-programming/min_cost_climbing_stairs.py` / `.cpp`）：

```python
def min_cost_climbing_stairs(cost):
    n = len(cost)
    if n <= 1:
        return 0
    prev2, prev1 = cost[0], cost[1]
    for i in range(2, n):
        prev2, prev1 = prev1, min(prev2, prev1) + cost[i]
    return min(prev2, prev1)
```

```cpp
int minCostClimbingStairs(const std::vector<int> &cost) {
    int n = static_cast<int>(cost.size());
    if (n <= 1) return 0;
    int prev2 = cost[0], prev1 = cost[1];
    for (int i = 2; i < n; ++i) {
        int cur = std::min(prev2, prev1) + cost[i];
        prev2 = prev1;
        prev1 = cur;
    }
    return std::min(prev2, prev1);
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：返回值是 `min(prev2, prev1)` 而不是 `prev1`——别忘了可以从倒数第二阶直接跨到
  顶；`dp[i]` 的定义里包含「支付了 `cost[i]`」，初值因此是 `cost[0]`、`cost[1]` 而非 0；
  只有一阶时返回 0（原地起步直接跨上去）。
- **相似题**：70. 爬楼梯（数方案 vs 求花费，见模式一）；198. 打家劫舍（把「每一步都要付」
  换成「选或不选」，见下）；64. 最小路径和（二维网格上的同类 `min` 递推）。

### 198. 打家劫舍（中等）

**题目**：沿街有一排房屋，每间藏有非负现金。相邻两间不能在同一个晚上都被闯入，否则报警。
求一夜之内能偷到的最高金额。

**思路**：

设 `dp[i]` 表示只考虑前 `i+1` 间房（下标 `0..i`）能偷到的最大金额。对第 `i` 间房，只有
两种互斥的选择：

- **偷**：那第 `i-1` 间就不能偷，收益是 `dp[i-2] + nums[i]`；
- **不偷**：收益就是 `dp[i-1]`。

取两者更大者：

```
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

初始化 `dp[-1] = dp[-2] = 0`（没有房子，收益为 0），从前往后推。只依赖前两项，滚动即可。

**为什么「偷」时要往回跳两间**：相邻约束说的正是「偷了 `i` 就不能偷 `i-1`」，所以前驱只能
是 `i-2`。这一跳把「相邻」这条规则准确翻译成了下标关系。

**为什么这是「决策型 DP」**：状态里没有连续增量，房子是离散的；核心是每一步在「做 / 不做」
之间权衡。很多经典的「选或不选」问题（打家劫舍、股票、背包）都是这个形状，区别只在约束。

**代码**（`src/dynamic-programming/house_robber.py` / `.cpp`）：

```python
def rob(nums):
    prev2, prev1 = 0, 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1
```

```cpp
int rob(const std::vector<int> &nums) {
    int prev2 = 0, prev1 = 0;
    for (int x : nums) {
        int cur = std::max(prev1, prev2 + x);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`prev2, prev1` 都从 0 起步，指向的是 `dp[-2]`、`dp[-1]`，所以循环可以从第一个
  元素直接开始，不必特判 `n = 0/1`；转移里是 `prev2 + x`（跳过相邻那间）而不是 `prev1 + x`；
  注意 Python 元组同时赋值与 C++ 临时变量的等价写法。
- **相似题**：213. 打家劫舍 II（把直线变成环，见下）；746. 使用最小花费爬楼梯（同为前驱二选一，
  但用 `min` 且每步必付，见上）；打家劫舍 III（树形 DP，在二叉树上做同样的选/不选）。

---

## 模式四：把「环」剪成「线」

**适用信号**：线性 DP 的规则都清楚，只是首尾之间多了一条约束，形成环。

**核心动作**：在环上选一处「剪开」，把问题展成线性，再枚举剪口两端**分别缺席**的少数几种
情况，各自跑一遍线性 DP 取最优。相邻约束下的环，剪在首尾之间恰好只有两种情况。

### 213. 打家劫舍 II（中等）

**题目**：房屋排成一圈，第一间和最后一间相邻。相邻两间不能同时被闯入，求最高金额。

**思路**：

和 198 只差一处：首尾相邻，因此**不能同时偷第一间和最后一间**。任何合法方案必然满足下面
两种情形之一：

- **不偷最后一间**：对 `nums[0..n-2]` 做一遍 198；
- **不偷第一间**：对 `nums[1..n-1]` 做一遍 198。

两种情形取最大值。因为「同时偷首尾」的方案本来就不合法，而其余方案一定不同时含首尾，所以
必定落在上述两种之一，不会漏解。

**为什么「剪环」是通法**：环形约束往往很难直接递推，但可以在某个位置剪一刀，把环展成线，
再枚举「这一刀两边谁缺席」的少数情况。这里剪在首尾之间，代价是跑两遍线性 DP，但仍然 O(n)。

**为什么单独处理只有一间房的情况**：`n = 1` 时没有「相邻」可言，直接返回该房金额；否则两个
子区间都会变成空数组、都返回 0，答案就错了。

**代码**（`src/dynamic-programming/house_robber_ii.py` / `.cpp`）：

```python
def rob(nums):
    if len(nums) == 1:
        return nums[0]

    def rob_range(lo, hi):
        prev2, prev1 = 0, 0
        for i in range(lo, hi):
            prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
        return prev1

    n = len(nums)
    return max(rob_range(0, n - 1), rob_range(1, n))
```

```cpp
int robRange(const std::vector<int> &nums, int lo, int hi) {
    int prev2 = 0, prev1 = 0;
    for (int i = lo; i < hi; ++i) {
        int cur = std::max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = cur;
    }
    return prev1;
}

int rob(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    return std::max(robRange(nums, 0, n - 1), robRange(nums, 1, n));
}
```

- **复杂度**：时间 O(n)（两趟线性扫描），空间 O(1)。
- **易错点**：两个区间的边界是 `[0, n-1)` 和 `[1, n)`，用左闭右开就能自然排除「最后一间」
  或「第一间」；`n == 1` 必须特判，别让它掉进「两个空区间」的分支；`n == 0` 在 C++ 里
  也要先挡掉，避免访问 `nums[0]`；`rob_range` 内部滚动逻辑与 198 完全一致，可复用。
- **相似题**：198. 打家劫舍（直线版，本题拆环后的两段都是它，见模式三）；环形数组的其它
  处理手法可对照 503. 下一个更大元素 II（环形数组遍历两遍取模，见栈篇）、918. 环形子数组
  的最大和。

---

## 模式五：以「我」结尾的连续段

**适用信号**：题目要的是「连续子数组 / 子串 / 子段」的最优值，连续性意味着候选区间由
左右两个端点确定。

**核心动作**：把状态钉在**右端点**上，定义 `dp[i]` 为「以第 i 个元素结尾的答案」。这样每个
连续段都有唯一结尾，枚举所有结尾就不会重复也不会遗漏。转移时只问一句：前一个结尾的最优
结果要不要接过来。

### 53. 最大子数组和（中等）

**题目**：给定整数数组 `nums`，找出一个具有最大和的连续子数组（至少一个元素），返回其最大和。

**思路**：

要求「连续」，就把它钉在结尾上。设 `dp[i] = 以 nums[i] 结尾的最大子数组和`。对当前元素，
只有两种选择：把前面以 `i-1` 结尾的最优段接上来，或者从自己重新开一段：

```
dp[i] = max(nums[i], dp[i-1] + nums[i])
```

前面那段的和 `dp[i-1]` 若为正，接上更划算；若为负，它就是拖累，不如撇掉从 `nums[i]` 另起。
因为只看前一项，用一个 `cur` 滚动即可。**但答案不是 `cur`**——最大子数组可能在任何位置结束，
所以再用一个 `best` 在每个位置顺手更新。

**为什么不能只返回最后一个状态**：`dp[i]` 的语义只保证「以 i 结尾」最大，并不保证全局最大。
这是「以结尾为状态」这类题共有的提醒：**结尾状态是过程，全局最优要另外维护。**

**代码**（`src/dynamic-programming/maximum_subarray.py` / `.cpp`）：

```python
def max_subarray_sum(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```

```cpp
int maxSubArray(const std::vector<int> &nums) {
    int best = nums[0], cur = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        cur = std::max(nums[i], cur + nums[i]);
        best = std::max(best, cur);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`best`、`cur` 都要用 `nums[0]` 初始化（题目保证至少一个元素，不能从 0 起步，
  否则全负数会错）；转移是 `max(x, cur + x)`，不是 `max(cur, cur + x)`；数组只有一个元素时
  循环不执行，直接返回它。
- **相似题**：674. 最长连续递增序列、718. 最长重复子数组（同为「以结尾为状态」的连续段，
  见下）；152. 乘积最大子数组（负数会让最小值翻成大，需要同时维护最大/最小两个状态）；
  与 `divide-conquer` 篇的分治解（左右最优 + 跨中点）对照，体会两种思路的差异。

### 674. 最长连续递增序列（简单）

**题目**：给定未经排序的整数数组 `nums`，找出最长且连续递增的子序列的长度。

**思路**：

和 300 只差「连续」二字，状态同样钉在结尾：`dp[i] = 以 nums[i] 结尾的连续递增序列长度`。
此时前驱只有一个——紧挨着的前一个元素：

```
dp[i] = dp[i-1] + 1   若 nums[i] > nums[i-1]
dp[i] = 1             否则（递增在 i 处断开，从 i 重新开始）
```

只依赖前一项，滚动变量即可。**和 300 对照**：不连续时要枚举前面所有更小的 `j`，是 O(n²)；
连续时前驱唯一，降到 O(n)。「连续」这个约束，往往能把枚举前驱的成本压掉。

**代码**（`src/dynamic-programming/longest_continuous_increasing_subsequence.py` / `.cpp`）：

```python
def find_length_of_lcis(nums):
    if not nums:
        return 0
    best = cur = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best
```

```cpp
int findLengthOfLCIS(const std::vector<int> &nums) {
    if (nums.empty()) return 0;
    int best = 1, cur = 1;
    for (size_t i = 1; i < nums.size(); ++i) {
        if (nums[i] > nums[i - 1]) {
            ++cur;
        } else {
            cur = 1;
        }
        best = std::max(best, cur);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：空数组返回 0，别让 `best = 1` 的初值泄漏；用「严格大于」判断递增（题目要
  `nums[i] > nums[i-1]`）；断链时 `cur` 必须重置为 1 而不是 0。
- **相似题**：300. 最长递增子序列（去掉「连续」，前驱从唯一变成任意，见下）；53. 最大子数组和
  （结构相同，只是聚合对象从「长度」换成「和」，见上）。

### 718. 最长重复子数组（中等）

**题目**：给两个整数数组 `nums1` 和 `nums2`，返回两个数组中公共的、长度最长的连续子数组的长度。

**思路**：

两个数组、又要求连续，把「以结尾为状态」直接升到二维：

```
dp[i][j] = nums1 以第 i 个元素结尾、nums2 以第 j 个元素结尾的最长公共后缀长度
```

只有两个结尾相等时这段公共后缀才有意义，且可以在去掉这两个结尾的基础上再接一格：

```
nums1[i-1] == nums2[j-1]  ->  dp[i][j] = dp[i-1][j-1] + 1
否则                      ->  dp[i][j] = 0
```

**和 1143 的关键区别**：这里不相等时**不能**从 `dp[i-1][j]` / `dp[i][j-1]` 继承，因为公共
子数组必须连续，一旦结尾对不上，以这对位置结尾的公共后缀长度只能归零。答案取整张表的最大值，
而不是 `dp[m][n]`。

**代码**（`src/dynamic-programming/maximum_length_of_repeated_subarray.py` / `.cpp`）：

```python
def find_length(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best
```

```cpp
int findLength(const std::vector<int> &nums1, const std::vector<int> &nums2) {
    int m = static_cast<int>(nums1.size());
    int n = static_cast<int>(nums2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                best = std::max(best, dp[i][j]);
            }
        }
    }
    return best;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（滚动数组可降到 O(n)）。
- **易错点**：`dp` 开 `(m+1) × (n+1)` 且下标偏移一位，`dp[0][*]`、`dp[*][0]` 是空后缀的
  0；不相等时保持 0，绝不能写 `max(dp[i-1][j], dp[i][j-1])`（那是 1143 的写法，会破坏
  连续性）；答案是全表最大值。
- **相似题**：1143. 最长公共子序列（不要求连续，不相等时可以继承邻格，见下）；674. 最长连续
  递增序列（单数组版，见上）；718 用滚动数组优化时要注意从右往左更新 `j`。

---

## 模式六：不要求连续的子序列

**适用信号**：题目说「子序列」，明确元素可以不连续、只要保持相对顺序。

**核心动作**：不连续意味着前驱不再唯一——单序列里可以是**前面任意一个**满足条件的元素，
双序列里可以是**跳过某一端的某个字符**。于是要么枚举前驱（单序列 O(n²)），要么用二维表把
「两个前缀的答案」记下来（双序列 O(m·n)）。

### 300. 最长递增子序列（中等）

**题目**：给定整数数组 `nums`，找到其中最长严格递增子序列的长度（子序列不要求连续）。

**思路**：

不连续时不能只盯前一个。那就反过来问：**以当前元素结尾**的最长递增子序列能有多长？设
`dp[i] = 以 nums[i] 结尾的最长递增子序列长度`。它至少是 1（只有自己）；只要前面有比
`nums[i]` 小的元素 `nums[j]`，都可以把以 `j` 结尾的最优解接过来：

```
dp[i] = 1 + max(dp[j])   对所有 j < i 且 nums[j] < nums[i]
```

答案为 `max(dp)`。外层 i 从小到大，保证算 `dp[i]` 时所有 `j < i` 都已就绪。

**为什么可以「枚举前面任意一个」**：因为子序列不要求连续，`nums[i]` 接在哪个较小的元素后面
都合法，所以要把所有可行前驱都试一遍取最大——这正是 O(n²) 的来源，也是和 674 的分水岭。

**代码**（`src/dynamic-programming/longest_increasing_subsequence.py` / `.cpp`）：

```python
def length_of_lis(nums):
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

```cpp
int lengthOfLIS(const std::vector<int> &nums) {
    if (nums.empty()) return 0;
    std::vector<int> dp(nums.size(), 1);
    int best = 1;
    for (size_t i = 0; i < nums.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (nums[j] < nums[i]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
        best = std::max(best, dp[i]);
    }
    return best;
}
```

- **复杂度**：时间 O(n²)，空间 O(n)。另有贪心 + 二分的 O(n log n) 解法（维护每个长度对应的
  最小结尾，用二分替换），但把状态藏进了数据结构，这里只作了解。
- **易错点**：条件是严格小于 `<`（题目要求严格递增），写成 `<=` 会算成非递减；`dp` 全部初始
  为 1；答案是 `max(dp)` 不是 `dp[-1]`；空数组返回 0。
- **相似题**：674. 最长连续递增序列（加上「连续」后前驱唯一、降到 O(n)，见上）；354. 俄罗斯
  套娃信封（先排序再对第二维做 LIS）；1143. 最长公共子序列（换成双序列，见下）。

### 1143. 最长公共子序列（中等）

**题目**：给定两个字符串 `text1`、`text2`，返回它们最长公共子序列的长度（不要求连续）；无公共
子序列返回 0。

**思路**：

两个字符串、可以跳字符，用二维表记录「两个前缀的答案」：

```
dp[i][j] = text1 的前 i 个字符与 text2 的前 j 个字符的最长公共子序列长度
```

只看两个前缀各自的最后一个字符：

- **相等**：它们一定可以配成公共子序列的末尾，接在各自去掉一个字符之后：

  ```
  dp[i][j] = dp[i-1][j-1] + 1
  ```

- **不等**：二者不可能同时作为末尾，至少舍弃一个，取两种舍弃里更优的：

  ```
  dp[i][j] = max(dp[i-1][j], dp[i][j-1])
  ```

边界 `dp[0][*] = dp[*][0] = 0`（一边为空，公共长度为 0）。两个方向都从小到大填，答案 `dp[m][n]`。

**为什么不等时要取两个方向的最大**：`text1[i-1]` 和 `text2[j-1]` 谁能派上用场是个选择，
跳过 `text1` 的末尾（`dp[i-1][j]`）和跳过 `text2` 的末尾（`dp[i][j-1]`）都有可能，所以要都看，
不能只沿对角线走。

**代码**（`src/dynamic-programming/longest_common_subsequence.py` / `.cpp`）：

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

```cpp
int longestCommonSubsequence(const std::string &text1, const std::string &text2) {
    int m = static_cast<int>(text1.size());
    int n = static_cast<int>(text2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)，滚动时注意保存左上角旧值）。
- **易错点**：`dp` 尺寸是 `(m+1)×(n+1)`，字符串下标要减 1；不等时忘记取 `max` 会漏掉跳过
  某一端的可能；答案就是右下角 `dp[m][n]`，不需要再取全表最大。
- **相似题**：718. 最长重复子数组（要求连续，不等时状态归零，见上）；72. 编辑距离（同一张
  表，把「相等继承 / 不等取 max」换成带代价的 min，见下）；583. 两个字符串的删除操作、
  1035. 不相交的线（都是 LCS 的换皮）。

---

## 模式七：回文类区间问题

**适用信号**：题目围绕「回文」——一个子串正着读反着读相同，或要求把字符串切分成回文段。

**核心动作**：回文的判定天然是从中心向两侧或从两端向中间收缩，所以有两条路：**中心扩展**
（枚举 2n-1 个中心向两边长，时间 O(n²)、空间 O(1)）和**区间 DP**（`dp[i][j]` 表示
`s[i..j]` 是否回文，由 `s[i]==s[j]` 且 `dp[i+1][j-1]` 推出，时间 O(n²)、空间 O(n²)）。

### 5. 最长回文子串（中等）

**题目**：给定字符串 `s`，找到 `s` 中最长的回文子串。

**思路**：

回文有一个「中心」，从中心向两边对称展开时字符始终相等。中心分两种：

- **奇数长度**：中心是一个字符，如 `"aba"` 的中心是 `'b'`；
- **偶数长度**：中心是两个字符之间的空隙，如 `"abba"` 的中心在 `'b'` 与 `'b'` 之间。

枚举全部 `2n-1` 个中心（`n` 个字符 + `n-1` 个空隙），各自向两边扩展到不能扩展，记录最长的一段。
代码里 `expand(i, i)` 处理奇中心，`expand(i, i+1)` 处理偶中心。

**为什么这里详展中心扩展而不是区间 DP**：两者时间同为 O(n²)，但中心扩展只用 O(1) 额外空间，
且「向两边长」的物理图像比填表更直观。区间 DP 的价值在需要复用回文判定的题（如 131 分割
回文串、516 最长回文子序列），这里只作一句话了解。

**代码**（`src/dynamic-programming/longest_palindromic_substring.py` / `.cpp`）：

```python
def longest_palindrome(s):
    if not s:
        return ""

    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    start, end = 0, 0
    for i in range(len(s)):
        l1, r1 = expand(i, i)
        l2, r2 = expand(i, i + 1)
        if r1 - l1 > end - start:
            start, end = l1, r1
        if r2 - l2 > end - start:
            start, end = l2, r2
    return s[start:end + 1]
```

```cpp
std::string longestPalindrome(const std::string &s) {
    if (s.empty()) return "";
    int n = static_cast<int>(s.size());

    auto expand = [&](int left, int right) {
        while (left >= 0 && right < n && s[left] == s[right]) {
            --left;
            ++right;
        }
        return std::make_pair(left + 1, right - 1);
    };

    int start = 0, end = 0;
    for (int i = 0; i < n; ++i) {
        auto odd = expand(i, i);
        auto even = expand(i, i + 1);
        if (odd.second - odd.first > end - start) {
            start = odd.first;
            end = odd.second;
        }
        if (even.second - even.first > end - start) {
            start = even.first;
            end = even.second;
        }
    }
    return s.substr(start, end - start + 1);
}
```

- **复杂度**：时间 O(n²)（每个中心最多扩展 O(n)，共 O(n) 个中心），空间 O(1)。
- **易错点**：必须同时枚举奇、偶两类中心，漏掉偶中心就找不出 `"bb"`；`expand` 退出时
  `left`、`right` 已越界一格，返回时要 `left+1, right-1`；空串要先挡掉。
- **相似题**：516. 最长回文子序列（不要求连续，用区间 DP，见 M-4）；647. 回文子串（数回文
  子串个数，中心扩展同样是标准解）；131. 分割回文串（见回溯篇，回文判定可复用本节）。

---

## 模式八：两个前缀之间的距离

**适用信号**：把一串字符改成另一串，问最少几步操作（增、删、改），或任何「两个前缀之间的
最小代价」问题。

**核心动作**：`dp[i][j]` 表示「第一个串前 i 个字符」变成「第二个串前 j 个字符」的最少代价。
两个结尾相等时零代价继承对角；不等时把三种操作分别翻译成「去掉某个结尾」的子问题，取最小。
边界就是「一方为空」的语义。

### 72. 编辑距离（困难）

**题目**：给两个单词 `word1`、`word2`，返回将 `word1` 转换成 `word2` 的最少操作数。允许
插入、删除、替换各一个字符。

**思路**：

又是双序列，但每步都有代价，用 DP 记前缀之间的距离：

```
dp[i][j] = 把 word1 前 i 个字符变成 word2 前 j 个字符的最少操作数
```

看两个前缀的最后一个字符：

- **相等**：这一步不用操作，问题缩小到去掉这两个字符，`dp[i][j] = dp[i-1][j-1]`。
- **不等**：三种操作各对应一个「消掉某个结尾」的子问题，取最小再加 1：
  - **删除** `word1[i-1]`：`dp[i-1][j]`（word1 少一个字符，仍要匹配 word2 的前 j 个）；
  - **插入** `word2[j-1]`：`dp[i][j-1]`（word2 少一个待匹配字符，word1 不变）；
  - **替换** `word1[i-1]` 为 `word2[j-1]`：`dp[i-1][j-1]`（两个结尾一起消掉）。

  ```
  dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
  ```

边界是「一方为空」的语义：`dp[i][0] = i`（把前 i 个删空要删 i 次）、`dp[0][j] = j`（从空串
插入 j 个字符要插 j 次）。答案 `dp[m][n]`。

**为什么插入对应 `dp[i][j-1]` 而不是别的**：把 word1 变成 word2 时，「插入 word2[j-1]」意味着
word2 还剩前 `j-1` 个字符没匹配，而 word1 一个都没消耗——所以是同一行往左挪一格。理解每个
操作对应哪个方向，是这道题的关键，也是它常被拿来当双序列 DP 模板的原因。

**代码**（`src/dynamic-programming/edit_distance.py` / `.cpp`）：

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
```

```cpp
int minDistance(const std::string &word1, const std::string &word2) {
    int m = static_cast<int>(word1.size());
    int n = static_cast<int>(word2.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
            }
        }
    }
    return dp[m][n];
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
- **易错点**：边界必须显式初始化成 `i` / `j`，不能留 0；相等时**不加 1**，直接继承对角；
  不等时是三项取 min 再 `+1`，容易漏掉「替换」那一项；C++ 用 `std::min({a,b,c})` 需要
  `#include <algorithm>`。
- **相似题**：1143. 最长公共子序列（同一张表，去掉了操作代价，可看作它的带权版，见上）；
  583. 两个字符串的删除操作（只允许删除，`dp` 变成两串长度和减 2·LCS）；10. 正则表达式
  匹配、44. 通配符匹配（双序列匹配的进阶，转移随模式字符展开）。

---

## 模式九：完全背包（物品可重复取）

**适用信号**：有一批「物品」，每件可以取**无限多次**，要凑出一个目标容量，问最少件数、
方案数或可行性。

**核心动作**：`dp[i]` 表示「凑出容量 i 的答案」，外层遍历物品、内层遍历容量且**容量正序**。
正序意味着 `dp[i - coin]` 可能已经用过当前这件物品，正好允许重复取——这就是完全背包的
标志。

### 322. 零钱兑换（中等）

**题目**：给你硬币面额数组 `coins` 和总金额 `amount`，返回凑成总金额所需的**最少硬币
个数**；凑不出返回 `-1`。每种硬币数量无限。

**思路**：

设 `dp[i]` 表示凑出金额 `i` 所需的最少硬币数。任何凑出 `i` 的方案，其**最后一枚硬币**
必然是某个面额 `coin`；去掉它，就得到一个凑出 `i - coin` 的方案。枚举所有可能的最后一枚
硬币取最小：

```
dp[i] = min(dp[i - coin] + 1)     对所有 coin <= i
```

初始化 `dp[0] = 0`（凑 0 元用 0 枚），其余置为一个「比任何可行解都大」的哨兵。
这里用 `amount + 1`——因为最多也只会用 `amount` 枚 1 元硬币。若终值仍是哨兵，说明凑不出，
返回 `-1`。

**为什么可用「最后一枚硬币」来分类**：一枚硬币作为「最后放的」是唯一确定的，去掉它方案就
缩短一截。这样按「末尾物品」分类，既不重复也不遗漏，正是背包型 DP 的通用切法。

**为什么容量要正序**：正序时 `dp[i - coin]` 已经是**本轮更新过**的值，等价于「这件物品
可以再取一次」，故允许重复使用。若倒序，每件物品至多取一次，就退化成 0/1 背包了。

**为什么内层可以直接从 `coin` 开始**：容量小于 `coin` 时这枚硬币根本放不进去，`dp[i - coin]`
会越界，跳过即可。

**代码**（完整可运行版见 `src/dynamic-programming/coin_change.py` / `.cpp`）：

```python
def coin_change(coins, amount):
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return -1 if dp[amount] == INF else dp[amount]
```

```cpp
int coinChange(const std::vector<int> &coins, int amount) {
    const int INF = amount + 1;
    std::vector<int> dp(amount + 1, INF);
    dp[0] = 0;
    for (int coin : coins) {
        for (int i = coin; i <= amount; ++i) {
            dp[i] = std::min(dp[i], dp[i - coin] + 1);
        }
    }
    return dp[amount] == INF ? -1 : dp[amount];
}
```

- **复杂度**：时间 O(amount × len(coins))，空间 O(amount)。
- **易错点**：哨兵别用 `INT_MAX`，`+1` 会溢出成负数（用 `amount + 1` 安全）；容量必须
  **正序**才能重复取物；最后要判哨兵，否则凑不出时返回的是哨兵值而不是 `-1`。
- **相似题**：279. 完全平方数（面额固定成平方数的完全背包，见下）；518. 零钱兑换 II
  （同型但求方案数，换聚合运算符，见下）；377. 组合总和 IV（同型求和但**区分顺序**，
  物品与容量循环位置调换，见对照）。

### 518. 零钱兑换 II（中等）

**题目**：给你硬币面额数组 `coins` 和总金额 `amount`，返回凑成总金额的**组合数**。
硬币无限，顺序不同的序列视为同一种组合。

**思路**：

状态与 322 完全一样：`dp[i]` 表示凑出金额 `i` 的方案数。区别只在**聚合运算符**——322 求
「最少几枚」用 `min`，这里求「有几种凑法」用**加法**：

```
dp[i] += dp[i - coin]     对所有 coin <= i
```

初始化 `dp[0] = 1`：凑出 0 元有且只有一种方案——什么都不选。**这里必须是 1 而不是 0**，
否则所有方案数都会是 0。

**为什么物品必须放在外层**：外层每固定一枚硬币，就把这枚硬币「一次性铺满」整行容量。这样
一个组合里硬币出现的先后顺序被强制为 `coins` 的下标顺序，于是「1+2」和「2+1」只会被算
作一种。若把容量放外层、硬币放内层，就会把不同顺序当成不同组合，得到的是**排列数**
（377. 组合总和 IV）。一句话：**求组合数——物品在外层；求排列数——容量在外层。**

**代码**（`src/dynamic-programming/coin_change_ii.py` / `.cpp`）：

```python
def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]
```

```cpp
int change(int amount, const std::vector<int> &coins) {
    std::vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int coin : coins) {
        for (int i = coin; i <= amount; ++i) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}
```

- **复杂度**：时间 O(amount × len(coins))，空间 O(amount)。
- **易错点**：`dp[0] = 1` 漏写或写成 0，方案数全变 0；物品与容量两层循环写反，组合数就
  变成了排列数；容量仍须正序以允许重复取物。
- **相似题**：322. 零钱兑换（同表同循环、只把 `min` 换成 `+=`，见上）；494. 目标和
  （同样是「数方案」的 0/1 背包，见模式十）；377. 组合总和 IV（把两层循环调换得到
  排列数，最典型的对照题）。

### 279. 完全平方数（中等）

**题目**：给你整数 `n`，返回和为 `n` 的完全平方数的**最少数量**（完全平方数如
1、4、9、16）。

**思路**：

把 1², 2², 3², … 看作面额，每种可重复使用，要凑出容量 `n` 且件数最少——这正是与 322
同型的完全背包，「硬币面额」被固定成了平方数。设 `dp[i]` 表示凑出 `i` 所需的最少平方数
个数，最后一件事物是某个平方数 `j²`：

```
dp[i] = min(dp[i - j²] + 1)     对所有 j² <= i
```

初始化 `dp[0] = 0`，其余先置为 `i`（用 `i` 个 1 一定能凑出，作为上界），再逐一取 `min`。

**为什么内层枚举到 `sqrt(i)` 即可**：这等价于把「平方数物品」逐个尝试一遍。它与「先物品
后容量」的完全背包写法是同一件事，只是这里物品是自然数平方、无需预先列成数组。

**代码**（`src/dynamic-programming/perfect_squares.py` / `.cpp`）：

```python
def num_squares(n):
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = i
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]
```

```cpp
int numSquares(int n) {
    std::vector<int> dp(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        dp[i] = i;
        for (int j = 1; j * j <= i; ++j) {
            dp[i] = std::min(dp[i], dp[i - j * j] + 1);
        }
    }
    return dp[n];
}
```

- **复杂度**：时间 O(n√n)，空间 O(n)。
- **易错点**：上界写成 `dp[i] = i` 是给「全用 1」兜底，漏掉可能出现非法的大数；内层
  条件用 `j * j <= i` 而非 `j <= i`，否则会访问 `dp` 的负下标（Python 会绕回末尾，静默
  出错）。
- **相似题**：322. 零钱兑换（同一副「完全背包求最少件数」的骨架，见上）；279 也可用
  「四平方和定理」在 O(√n) 内判定，但通用背包解更值得先掌握。

---

## 模式十：0/1 背包（每件至多取一次）

**适用信号**：从一批物品里挑选，每件**只能选一次**，判断能否凑出某目标和、有多少种选法，
或凑得离目标多近。

**核心动作**：与完全背包只差一个字——**容量倒序**。倒序保证 `dp[i - num]` 用的是「本轮
之前」的旧值，从而同一件物品不会被选两次。

### 416. 分割等和子集（中等）

**题目**：给你只含正整数的非空数组 `nums`，判断能否把它分割成两个子集，使两个子集的
元素和相等。

**思路**：

「两个子集和相等」等价于「存在一个子集的和恰好是整个数组和的一半」。设总和为 `total`，
若 `total` 为奇数则直接返回 `False`；否则问题变成：能否从 `nums` 中挑出若干个数，使其和
恰为 `target = total // 2`。

设 `dp[i]` 表示「能否选出和为 `i` 的子集」。每遇到一个数 `num`，它要么不选、要么选，于是：

```
dp[i] = dp[i] or dp[i - num]     i 从 target 递减到 num
```

初始化 `dp[0] = True`（和为 0 的子集就是空集），答案 `dp[target]`。

**为什么容量要倒序**：正序时 `dp[i - num]` 可能在本轮刚被更新过，等于同一个 `num` 被用
了两次；倒序保证用到的是「本轮之前」的旧值，于是每件物品至多用一次。这就是 0/1 背包
「一维数组倒序」的全部来历。

**代码**（`src/dynamic-programming/partition_equal_subset_sum.py` / `.cpp`）：

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]
```

```cpp
bool canPartition(const std::vector<int> &nums) {
    int total = 0;
    for (int x : nums) total += x;
    if (total % 2 != 0) return false;
    int target = total / 2;
    std::vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int num : nums) {
        for (int i = target; i >= num; --i) {
            dp[i] = dp[i] || dp[i - num];
        }
    }
    return dp[target];
}
```

- **复杂度**：时间 O(n × target)，空间 O(target)。
- **易错点**：容量**倒序**是一票否决的错误点，正序会退化成完全背包得出错误结论；先判
  总和奇偶，奇数直接 `False`；`dp[0] = True` 不能少。
- **相似题**：494. 目标和（同样是 0/1 背包，但数方案，见下）；1049. 最后一块石头的重量 II
  （不要求恰好一半，而是取最接近一半，见下）；1049 与本题的差别正是「恰好」与「最接近」。
- **一句话多解**：这题也可用 2D DP `dp[i][j]` 写，压成一维后就是上面的倒序循环；`BitSet`
  或 Python `int` 位运算同样可解，思路一致。

### 494. 目标和（中等）

**题目**：给你非负整数数组 `nums` 和整数 `target`，给每个数前添加 `+` 或 `-`，返回能
构造出运算结果等于 `target` 的不同表达式数目。

**思路**：

直接枚举每个数取正或取负是指数级。把式子改写：设取 `+` 的数之和为 `P`，取 `-` 的数之和
为 `N`（都非负），则

```
P - N = target,   P + N = total（数组总和）
```

两式相加得 `P = (total + target) / 2`。于是问题变成「选出一个和为 `P` 的子集，有多少种
选法」，即 0/1 背包数方案：

```
dp[i] += dp[i - num]     i 从 P 递减到 num
```

初始化 `dp[0] = 1`，答案 `dp[P]`。先判可行性：若 `(total + target)` 是奇数，或
`|target| > total`，直接返回 0。

**为什么能这样转化**：加号集合一旦确定，其余全是减号，整个表达式就唯一确定。一次转化
就把指数枚举降成多项式 DP，是这类题最漂亮的一步。

**代码**（`src/dynamic-programming/target_sum.py` / `.cpp`）：

```python
def find_target_sum_ways(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2 != 0:
        return 0
    p = (total + target) // 2
    dp = [0] * (p + 1)
    dp[0] = 1
    for num in nums:
        for i in range(p, num - 1, -1):
            dp[i] += dp[i - num]
    return dp[p]
```

```cpp
int findTargetSumWays(const std::vector<int> &nums, int target) {
    int total = 0;
    for (int x : nums) total += x;
    if (std::abs(target) > total || (total + target) % 2 != 0) return 0;
    int p = (total + target) / 2;
    std::vector<int> dp(p + 1, 0);
    dp[0] = 1;
    for (int num : nums) {
        for (int i = p; i >= num; --i) {
            dp[i] += dp[i - num];
        }
    }
    return dp[p];
}
```

- **复杂度**：时间 O(n × P)，空间 O(P)，其中 P = (total + target) / 2。
- **易错点**：忘记先判 `(total + target)` 的奇偶与 `|target| > total`，会得到错误的
  下标或答案；`dp[0] = 1` 不能少；容量倒序保证每件只用一次；`nums` 含 0 时，`dp[0] = 1`
  依然正确，因为 0 可以被选或不选，方案数会自然翻倍。
- **相似题**：416. 分割等和子集（同型可行性，见上）；1049. 最后一块石头的重量 II
  （同型取最接近，见下）；518. 零钱兑换 II（数方案但可重复取，注意与本题的循环顺序差异）。

### 1049. 最后一块石头的重量 II（中等）

**题目**：有一堆正整数重量的石头，每回合任选两块 `x <= y` 一起粉碎：若 `x == y` 两块都
消失，否则剩下一块 `y - x`。问最后可能剩下的**最小重量**。

**思路**：

把每次「粉碎」看成给每块石头分配一个正号或负号，最终重量就是带符号重量之和的绝对值。
于是问题化为：把石头分成两组，使两组重量尽量相等。

设总和为 `total`，找一个子集和 `s` 尽量接近 `total/2`，则两组之差 `total - 2s` 就是最小
剩余重量。容量上界取 `target = total // 2`，每块石头倒序更新（0/1 背包）：

```
dp[i] = dp[i] or dp[i - w]     i 从 target 递减到 w
```

最后从 `target` 向下找最大的可达和 `s`，返回 `total - 2 * s`。

**为什么与 416 不同**：416 问「能否**恰好**凑一半」，返回布尔；这里问「**最接近**一半是
多少」，于是最后要扫出最大的可行容量。状态与转移完全一样，只是答案的取法变了。

**代码**（`src/dynamic-programming/last_stone_weight_ii.py` / `.cpp`）：

```python
def last_stone_weight_ii(stones):
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for w in stones:
        for i in range(target, w - 1, -1):
            dp[i] = dp[i] or dp[i - w]
    for s in range(target, -1, -1):
        if dp[s]:
            return total - 2 * s
    return total
```

```cpp
int lastStoneWeightII(const std::vector<int> &stones) {
    int total = 0;
    for (int w : stones) total += w;
    int target = total / 2;
    std::vector<bool> dp(target + 1, false);
    dp[0] = true;
    for (int w : stones) {
        for (int i = target; i >= w; --i) {
            dp[i] = dp[i] || dp[i - w];
        }
    }
    for (int s = target; s >= 0; --s) {
        if (dp[s]) return total - 2 * s;
    }
    return total;
}
```

- **复杂度**：时间 O(n × total)，空间 O(total)。
- **易错点**：容量是 `total // 2` 而非 `total`（超过一半与镜像同解，浪费一半空间）；最后
  要从大到小扫第一个可达 `s`；容量倒序不可少；总和本身的奇偶不影响正确性，最终用
  `total - 2s` 算差即可。
- **相似题**：416. 分割等和子集（恰好一半 vs 最接近一半，见上）；494. 目标和（同为 0/1
  背包，一个数方案、一个求极值）；本题的「两块相消」正是 1046. 最后一块石头的重量（用
  大顶堆模拟）的升级版，那题按固定规则模拟、这题才需要做划分。

---

## 模式十一：把状态切成「前缀 + 尾段」

**适用信号**：一个字符串能否被字典中的词拼出来，或能否被某些分隔切成若干段，问可行性
（或最少段数）。

**核心动作**：`dp[i]` 表示「前 `i` 个字符能否被完整覆盖」。枚举最后一个切点 `j`：
若 `dp[j]` 为真且 `s[j:i]` 满足条件，则 `dp[i]` 为真。状态钉在**前缀末尾**，一刀切出
「前缀」和「尾段」。

### 139. 单词拆分（中等）

**题目**：给你字符串 `s` 和字符串列表 `wordDict`，若能用字典里的词（可重复使用）拼出
`s`，返回 `true`。

**思路**：

设 `dp[i]` 表示「`s` 的前 `i` 个字符能否被字典拼出」。考虑最后一段单词 `s[j:i]`，则：

```
dp[i] = true，当存在 j 使 dp[j] 为真且 s[j:i] 在字典中。
```

换句话说，前缀 `i` 可拼 ⟺ 它能被切在某个「可拼的前缀 `j`」之后，且剩下的一截本身是字典
词。初始化 `dp[0] = true`（空串当然可拼），答案 `dp[n]`。

**为什么状态盯住「前缀末尾」**：切割关心的是「前 `i` 个字符能否被完整覆盖」，末尾位置
天然是状态；每切一刀就把问题拆成「前缀」与「最后一截」。这与 53/674 那种「以 `i` 结尾」
的状态同源，只是判断条件换成了「查字典」。

**为什么用集合**：内层反复判断 `s[j:i]` 是否在字典，把 `wordDict` 转成 `set` 后判断是
O(1)（不计子串哈希本身的开销）。

**代码**（`src/dynamic-programming/word_break.py` / `.cpp`）：

```python
def word_break(s, word_dict):
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]
```

```cpp
bool wordBreak(const std::string &s, const std::vector<std::string> &wordDict) {
    std::unordered_set<std::string> words(wordDict.begin(), wordDict.end());
    int n = static_cast<int>(s.size());
    std::vector<bool> dp(n + 1, false);
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (dp[j] && words.count(s.substr(j, i - j))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}
```

- **复杂度**：时间 O(n²)（子串哈希本身是 O(n)，严格说更高；可加「最长词长」上界优化把
  内层收窄），空间 O(n)。
- **易错点**：`dp[0] = true` 是空串这个 base case，漏了整题全假；内层一旦命中即可
  `break`；空串返回 `true`（可拼）。
- **相似题**：140. 单词拆分 II（在本题基础上回溯输出所有句子，见回溯篇）；91. 解码方法、
  132. 分割回文串 II（都是「按前缀切分」的 DP，把判定条件换成解码合法 / 回文即可）；
  131. 分割回文串（同一切割框架的回溯版，见回溯篇）。

---

## 模式十二：网格上的二维 DP

**适用信号**：问题的舞台本身就是一张 `m × n` 的网格，每一步只能向下或向右推进；
或者答案与「某个格子所在位置」强绑定，需要知道走到它的累计代价 / 方案数。

**核心动作**：状态设成 `dp[i][j]`，含义是「走到 `(i, j)` 的答案」或「以 `(i, j)` 为
右下角的答案」。转移只从**上方** `dp[i-1][j]`、**左方** `dp[i][j-1]` 聚合而来；遇到
「正方形」这类问题还要多看一个**左上** `dp[i-1][j-1]`。运算符仍然是问题的问法：**数方案
用加法（+），求最小代价用 `min`**。最后别忘了：有障碍物的格子相当于「挖了个洞」，
对应状态直接置 0。

### 62. 不同路径（中等）

**题目**：一个机器人位于 `m × n` 网格的左上角，每次只能**向下或向右**移动一步，问到达
右下角共有多少条不同的路径。

**思路**：

设 `dp[i][j]` 表示从左上角走到第 `i` 行第 `j` 列的路径数。机器人只能从**上方**
`(i-1, j)` 或**左方** `(i, j-1)` 进入当前格，所以走到 `(i, j)` 的方案数就是这两条来路的
方案数之和：

```
dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

第一行和第一列要单独想清楚：站在第一行的任意格子，只能一路向右走过来，只有这一种走法，
所以它们全是 1。注意这不是「没有前驱」，而是「前驱唯一」——把它当 1 而不是 0，答案才对。

**代码**（`src/dynamic-programming/unique_paths.py` / `.cpp`）：

```python
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

```cpp
int uniquePaths(int m, int n) {
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 1));
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（可只留一行，滚动到 O(n)）。
- **易错点**：第一行/第一列要初始化为 1，不能留 0；`dp` 定义成 `(m+1)×(n+1)` 时下标要
  相应偏移，但最干净的做法就是 `m×n` 全 1 起步。
- **相似题**：63. 不同路径 II（下面加障碍物）；64. 最小路径和（同一张网格换成求 `min`）；
  62 还有个数学解 `C(m+n-2, m-1)`，但 DP 更能迁移到有障碍、有权重的变体。

### 63. 不同路径 II（中等）

**题目**：在 `m × n` 网格中，`1` 表示障碍物、`0` 表示空地。机器人同样从左上角出发、
只能向下或向右，求到达右下角的路径数（路径上不能有障碍物）。

**思路**：

转移与 62 完全相同，只是加一条规则：**若 `(i, j)` 是障碍物，则 `dp[i][j] = 0`**。
因为没有任何路径能停在障碍物上，所有经过它的方案自然断掉。

写成「先判断当前格是不是障碍物，再把上方、左方的值加进来」，就不必单独处理「第一行/
第一列被障碍物截断」的麻烦情况：障碍物之后的格子会因为前驱是 0 而自动变成 0。
起始格本身是障碍物时，直接返回 0。

**代码**（`src/dynamic-programming/unique_paths_ii.py` / `.cpp`）：

```python
def unique_paths_with_obstacles(obstacle_grid):
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    if obstacle_grid[0][0] == 1:
        return 0
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                dp[i][j] = 0
                continue
            if i > 0:
                dp[i][j] += dp[i - 1][j]
            if j > 0:
                dp[i][j] += dp[i][j - 1]
    return dp[m - 1][n - 1]
```

```cpp
int uniquePathsWithObstacles(const std::vector<std::vector<int>> &obstacleGrid) {
    int m = static_cast<int>(obstacleGrid.size());
    int n = static_cast<int>(obstacleGrid[0].size());
    if (obstacleGrid[0][0] == 1) return 0;
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    dp[0][0] = 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (obstacleGrid[i][j] == 1) {
                dp[i][j] = 0;
                continue;
            }
            if (i > 0) dp[i][j] += dp[i - 1][j];
            if (j > 0) dp[i][j] += dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
- **易错点**：障碍物格必须显式置 0 并 `continue`，否则会把「停在障碍物上」的方案也加进去；
  起始格是障碍物要提前返回；`dp[0][0] = 1` 只有在起始格不是障碍物时才成立。
- **相似题**：62. 不同路径（无障碍版）；64. 最小路径和（若格子带代价，障碍物可理解为
  「不可通行」的大代价）；980. 不同路径 III（要经过所有空格，转为回溯 / 状态压缩）。

### 64. 最小路径和（中等）

**题目**：给一个 `m × n` 的**非负**整数网格 `grid`，从左上角走到右下角、每次只能向下或
向右，求路径上数字之和的**最小值**。

**思路**：

网格的走法没变，只是把「数方案」换成「求最小代价」。设 `dp[i][j]` 为从左上角走到
`(i, j)` 的最小路径和，到达它的最后一步只能来自上方或左方，再加上当前格的代价：

```
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
```

边界同样是第一行、第一列只能沿一条直线走过来，直接累加。对比 62 可以清楚看到：
**网格、遍历方式、依赖方向都一模一样，唯一变的是运算符（`+` vs `min`）和要不要加上
当前格权重**——这正是「运算符即问题问法」的又一例证。

**代码**（`src/dynamic-programming/minimum_path_sum.py` / `.cpp`）：

```python
def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[m - 1][n - 1]
```

```cpp
int minPathSum(const std::vector<std::vector<int>> &grid) {
    int m = static_cast<int>(grid.size());
    int n = static_cast<int>(grid[0].size());
    std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
    dp[0][0] = grid[0][0];
    for (int i = 1; i < m; ++i) dp[i][0] = dp[i - 1][0] + grid[i][0];
    for (int j = 1; j < n; ++j) dp[0][j] = dp[0][j - 1] + grid[0][j];
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[i][j] = grid[i][j] + std::min(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[m - 1][n - 1];
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
- **易错点**：第一行、第一列要逐格累加，不能简单置 1；转移里**别忘了加 `grid[i][j]`**，
  否则算的是「只累计起点」的错误值；`dp[0][0]` 是 `grid[0][0]` 而非 0。
- **相似题**：62/63 不同路径（同网格求方案数）；174. 地下城游戏（从右下往左上倒推，且
  要保证中途血量不为负，转移更绕）；931. 下降路径最小和（每一步可向左下/正下/右下走，
  是「多一个方向」的网格 DP）。

### 221. 最大正方形（中等）

**题目**：在一个由 `'0'` 和 `'1'` 组成的二维矩阵中，找出只包含 `'1'` 的最大正方形，
返回它的面积。

**思路**：

还是网格 DP，但状态换了个更巧的定义：**`dp[i][j]` 表示以 `(i, j)` 为右下角、且全部由
`'1'` 构成的最大正方形的边长**。若当前格是 `'1'`，它能向上方、左方、左上方三块正方形
「借」边长，取三者中最短的那个再 +1：

```
dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
```

为什么取最小：以 `(i,j)` 为右下角、边长为 `k` 的正方形，其上方、左方、左上方必须各自
存在一块至少 `k-1` 的正方形；反过来说，三者中最短的那块决定了当前能长到多大——**短板
决定边长**。若当前格是 `'0'`，则 `dp[i][j] = 0`。全程记录最大边长，答案就是它的平方。

**代码**（`src/dynamic-programming/maximal_square.py` / `.cpp`）：

```python
def maximal_square(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == "1":
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                best = max(best, dp[i][j])
    return best * best
```

```cpp
int maximalSquare(const std::vector<std::vector<char>> &matrix) {
    if (matrix.empty() || matrix[0].empty()) return 0;
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    int best = 0;
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (matrix[i - 1][j - 1] == '1') {
                dp[i][j] = std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                best = std::max(best, dp[i][j]);
            }
        }
    }
    return best * best;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（滚动到一维时，`dp[j-1]` 是左方、`dp[j]` 是上方，
  需要在覆盖前把左上角的旧值备份下来）。
- **易错点**：答案要返回**面积**（边长的平方）；`dp` 用 `(m+1)×(n+1)` 的哨兵行/列可以
  免去第一行第一列特判；比较的是字符 `'1'` 不是整数 1。
- **相似题**：1277. 统计全为 1 的正方形子矩阵（把「最大边长」改成「所有 `dp[i][j]` 求和」，
  因为每个 `dp[i][j]` 恰好是以 `(i,j)` 结尾的正方形个数）；85. 最大矩形（本题的「变高」
  版本，需要转成柱状图 + 单调栈，见栈篇）。

---

## 模式十三：区间 DP（枚举「最后一步」）

**适用信号**：答案依赖一段连续下标区间 `[i, j]` 内部怎么处理，并且这个区间能被「某个点」
一刀切成左右两半，两半的结果可以独立求解。典型的信号是「戳破 / 删除一个元素后左右合并」
「回文子串 / 子序列」。

**核心动作**：设 `dp[i][j]` 表示区间 `[i, j]`（有时是**开区间** `(i, j)`）上的最优解，
按**区间长度从小到大**填表。难点在于找到「最后一步」发生在区间的哪个位置：一旦确定最后
处理的元素 `k`，收益（或转移）往往只与边界和 `k` 有关，左右两段则各自退化成更小的同型
子问题。回文类问题则是「看两端字符配不配」，配则缩进一格 +2，不配则丢一边取 `max`。

### 312. 戳气球（困难）

**题目**：有 `n` 个气球排成一排，第 `i` 个气球上写着 `nums[i]`。戳破第 `i` 个气球可得
`nums[i-1] * nums[i] * nums[i+1]` 枚硬币（越界的气球按 1 计算），戳破后它与左右气球
变成相邻。求能获得硬币的最大数量。

**思路**：

正向想很难：戳一个气球会改变左右邻居，后续依赖被破坏，状态没法定义。**反过来想——枚举
区间里最后一个被戳破的气球 `k`。** 当 `k` 被戳时，开区间 `(i, j)` 内除它以外的气球已经
全没了，而两侧边界 `i`、`j` 还留着（它们是被保留的哨兵），于是这一步的收益恰好是
`vals[i] * vals[k] * vals[j]`，且 `k` 左侧、右侧内部的戳法互不影响，可以独立求最优。

先在数组两端各加一个值为 1 的虚拟气球作为**永不戳破的边界哨兵**，得到 `vals`。
定义 `dp[i][j]` 为「戳光开区间 `(i, j)` 内所有气球」的最大收益（`i`、`j` 保持不动）：

```
dp[i][j] = max over k in (i, j) of
           vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j]
```

按区间长度从小到大填表，答案是 `dp[0][n+1]`（`n` 为原气球个数）。

**为什么「最后一个」能让子问题独立**：`k` 是最后戳的，意味着戳它时 `(i, j)` 里所有邻居
都已消失，因此它只与边界 `i`、`j` 有关；而 `k` 左边、右边内部各自的戳法是一段更小的
同型问题，不会跨越 `k`。这就是区间 DP 的通用套路：**把「顺序」反转成「最后一个」，
依赖就断开了。**

**代码**（`src/dynamic-programming/burst_balloons.py` / `.cpp`）：

```python
def max_coins(nums):
    vals = [1] + list(nums) + [1]
    n = len(vals)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                gain = vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j]
                if gain > dp[i][j]:
                    dp[i][j] = gain
    return dp[0][n - 1]
```

```cpp
int maxCoins(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<int> vals(n + 2, 1);
    for (int i = 0; i < n; ++i) vals[i + 1] = nums[i];
    int total = n + 2;
    std::vector<std::vector<int>> dp(total, std::vector<int>(total, 0));
    for (int length = 2; length < total; ++length) {
        for (int i = 0; i + length < total; ++i) {
            int j = i + length;
            for (int k = i + 1; k < j; ++k) {
                int gain = vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j];
                dp[i][j] = std::max(dp[i][j], gain);
            }
        }
    }
    return dp[0][total - 1];
}
```

- **复杂度**：时间 O(n³)（区间 O(n²) 个，每个枚举分割点 O(n)），空间 O(n²)。
- **易错点**：`dp` 用的是**开区间** `(i, j)`，`k` 只取 `i+1..j-1`；填表顺序必须按
  **区间长度**递增，否则用到的子区间还没算；两端补 1 的哨兵别忘，空输入时长度为 2 的
  区间 `(0,1)` 收益为 0 也自洽。
- **相似题**：1039. 多边形三角剖分的最低得分、1000. 合并石头的最低成本（都是「选定最后
  一次合并/分割，把区间一分为二」的区间 DP）；546. 移除盒子（区间 DP 再加一维「左边带
  了几个同色」）。

### 516. 最长回文子序列（中等）

**题目**：给一个字符串 `s`，找出其中最长的回文**子序列**的长度。子序列不要求连续，但
必须保持字符的相对顺序。

**思路**：

回文是典型的区间问题：设 `dp[i][j]` 表示子串 `s[i..j]` 中最长回文子序列的长度，看区间
两端：

- `s[i] == s[j]`：两端可以一起用上，拼在中间最优解的外面，`dp[i][j] = dp[i+1][j-1] + 2`；
- `s[i] != s[j]`：两端不可能同时用上（否则回文两端必须相同），于是丢弃其中一个，
  取两者较大：`dp[i][j] = max(dp[i+1][j], dp[i][j-1])`。

单字符区间是长度 1 的回文，`dp[i][i] = 1`。因为 `dp[i][j]` 依赖下一行 `i+1` 和本行左边
`j-1`，遍历时 `i` 从大到小、`j` 从小到大，保证用到的都算过了。

本题与最长回文**子串**（5，见模式七）的区别一句点破：子序列可以跳过字符，两端不等时是
「丢一边取 `max`」；而子串必须连续，只能用中心扩展 / 区间判真。

**代码**（`src/dynamic-programming/longest_palindromic_subsequence.py` / `.cpp`）：

```python
def longest_palindrome_subseq(s):
    n = len(s)
    if n == 0:
        return 0
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

```cpp
int longestPalindromeSubseq(const std::string &s) {
    int n = static_cast<int>(s.size());
    if (n == 0) return 0;
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = n - 1; i >= 0; --i) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = std::max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[0][n - 1];
}
```

- **复杂度**：时间 O(n²)，空间 O(n²)（可滚动到 O(n)）。
- **易错点**：`i` 必须**从大到小**遍历，因为要用到 `dp[i+1][...]`；相等时是 `+2` 不是
  `+1`（两端各贡献一个字符）；不等时是 `max` 而不是像最长公共子序列那样凭空多一步；
  空串要特判返回 0。
- **相似题**：647. 回文子串（下面，求个数而非长度）；5. 最长回文子串（连续版，中心扩展
  更快，见模式七）；1312. 让字符串成为回文串的最少插入次数（答案是 `n - 最长回文子序列`，
  几乎就是本题的改写）；1143. 最长公共子序列（把 `s` 与它的反转求 LCS 也等于本题答案，
  见模式六）。

### 647. 回文子串（中等）

**题目**：给一个字符串 `s`，统计它有多少个回文**子串**（不同位置算不同的子串）。

**思路**：

用布尔区间 DP：`dp[i][j]` 表示 `s[i..j]` 是否为回文。判定只看两件事——两端字符是否
相同、以及去掉两端后的中间段是否回文：

```
s[i] == s[j] 且 (区间长度 <= 2 或 dp[i+1][j-1])
```

长度不足 2 时中间是空串，天然回文，所以用 `j - i < 2` 兜底。每得到一个 `True` 就计数
加一。遍历顺序同样是 `i` 从大到小、`j` 从小到大，保证 `dp[i+1][j-1]` 已经算好。

本题也有一条更省空间的路线：**中心扩展**（枚举 `2n-1` 个中心向两边长，O(1) 额外空间），
与最长回文子串（5）同款。这里选择区间 DP，是为了和 516、312 共用同一个「按区间长度填表」
的框架，方便对照。

**代码**（`src/dynamic-programming/palindromic_substrings.py` / `.cpp`）：

```python
def count_substrings(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                dp[i][j] = True
                count += 1
    return count
```

```cpp
int countSubstrings(const std::string &s) {
    int n = static_cast<int>(s.size());
    std::vector<std::vector<bool>> dp(n, std::vector<bool>(n, false));
    int count = 0;
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            if (s[i] == s[j] && (j - i < 2 || dp[i + 1][j - 1])) {
                dp[i][j] = true;
                ++count;
            }
        }
    }
    return count;
}
```

- **复杂度**：时间 O(n²)，空间 O(n²)（中心扩展是 O(1) 空间）。
- **易错点**：`j` 从 `i` 开始（要覆盖长度为 1 的回文）；`j - i < 2` 这根分支不能少，
  否则访问 `dp[i+1][j-1]` 会越界或拿到未定义值；`i` 从大到小、`j` 从小到大。
- **相似题**：5. 最长回文子串（求最长而非计数，中心扩展，见模式七）；516. 最长回文子
  序列（不连续版）；132. 分割回文串 II（用回文判定 + 切割 DP 求最少分割数，见回溯 /
  切割篇）。

---

## 模式十四：状态机 DP（股票买卖）

**适用信号**：每天要做「买 / 卖 / 不动」的决策，而「今天能不能买、能不能卖」只取决于
手上有没有股票，或者再多几个附加状态（刚卖过、还能交易几次）。决策的收益逐日累加，要求
最终最大收益。

**核心动作**：把「手头的姿态」定义成有限个状态，每个状态记录**到达该姿态时的最大累计
收益**；每读到一天的价格，就对所有状态做一轮松弛（保持不动，或从别的状态切换过来）。
一天之内多个状态可以同时更新——因为同一天先卖后买是允许的。

最朴素的骨架只有两个状态：`hold`（持股）和 `cash`（空仓）。

```
hold = max(hold, cash - p)   # 保持持股，或用已实现的收益买入
cash = max(cash, hold + p)   # 保持空仓，或把股票卖出
```

整个买卖股票系列都是这套骨架：**121 只改一个入口（买入从 0 起算），122 放开次数，
123 / 188 把状态按「第几笔」摊开，309 多一个「刚卖出」的中间态，714 在卖出时扣手续费。**

### 121. 买卖股票的最佳时机（简单）

**题目**：给定价格数组 `prices`，只能选择某一天买入、之后某一天卖出，**最多完成一笔**
交易，求最大利润；不能获利返回 0。

**思路**：

把「手上有没有股票」当成两个状态，边走边维护各自的最大累计收益：

- `hold`：目前持有股票时的最大收益（负数表示已经花了多少钱买入）；
- `cash`：目前不持有股票时的最大收益。

每读到一个价格 `p`，两个状态各自可以「保持」或「切换」：

```
hold = max(hold, -p)          # 保持持股，或今天买入（第一笔，从 0 收益扣 p）
cash = max(cash, hold + p)    # 保持空仓，或今天把股票卖掉（加 p）
```

**为什么这样就能保证只买一次**：`hold` 的买入来源是常数 `-p`，而不是接在已实现收益
之上；也就是说 `hold` 里永远只含一笔买入的成本。把它卖掉得到 `cash`，恰好完成一笔完整
交易。空仓一定不比持仓差（可以卖掉落袋），所以返回 `cash`。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock.py` / `.cpp`）：

```python
def max_profit(prices):
    hold = float("-inf")
    cash = 0
    for p in prices:
        hold = max(hold, -p)
        cash = max(cash, hold + p)
    return cash
```

```cpp
int maxProfit(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, -static_cast<long long>(p));
        cash = std::max(cash, hold + p);
    }
    return static_cast<int>(cash);
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`hold` 初值要用负无穷（一开始没股票，不能持有）；买入那一步必须写
  `-p` 而不是 `cash - p`，否则会变成允许多笔（= 122）；最后的答案取 `cash` 而非
  `hold`（空仓落袋）。
- **相似题**：122（放开交易次数，买入改成从 `cash` 扣，见下）；买卖股票系列其余各题
  （见下）；本题也常用「记录历史最低价，边扫边算差价」的贪心写法，二者等价。

### 122. 买卖股票的最佳时机 II（简单）

**题目**：同样的价格数组，但可以完成**任意多笔**交易（买入前必须先卖掉手上的股票，
同一天可以先卖后买），求最大利润。

**思路**：

状态定义与 121 完全一样，仍是 `hold` 和 `cash`。区别只在**买入时的资金来源**：既然允许
反复交易，今天买入可以动用之前赚到的钱，于是从 `cash` 里扣：

```
hold = max(hold, cash - p)
cash = max(cash, hold + p)
```

`cash - p` 表示「用前面几笔赚到的收益，今天再买一只」。这样每一段上涨都能被完整吃下，
结果等价于「把所有相邻上涨的差额相加」。最后返回 `cash`。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock_ii.py` / `.cpp`）：

```python
def max_profit_ii(prices):
    hold = float("-inf")
    cash = 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p)
    return cash
```

```cpp
int maxProfitII(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, cash - static_cast<long long>(p));
        cash = std::max(cash, hold + p);
    }
    return static_cast<int>(cash);
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：与 121 只差买入一处（`-p` → `cash - p`），抄错就把无限笔退化成了一笔；
  同一天允许「先卖后买」，所以先更新 `hold` 再更新 `cash` 没有问题。
- **相似题**：121（限一笔，是本题的收紧版）；714（无限笔但每笔扣手续费，见下）；
  188（把次数设成参数 k，k 足够大时等价于本题）。

### 123. 买卖股票的最佳时机 III（困难）

**题目**：最多完成**两笔**交易，求最大利润。

**思路**：

一笔交易由「买入」「卖出」两个动作组成。最多两笔，就自然有四个阶段，每个阶段各设一个
状态记录其最大收益：

- `buy1`：第一笔已买入、尚未卖出；
- `sell1`：第一笔已完成；
- `buy2`：第二笔已买入、尚未卖出；
- `sell2`：第二笔已完成（答案）。

每读到一个价格 `p`，四个状态**依次**松弛——这个顺序天然满足「先买后卖、先卖后买」：

```
buy1  = max(buy1,  -p)          # 第一次买入
sell1 = max(sell1, buy1 + p)    # 第一次卖出
buy2  = max(buy2,  sell1 - p)   # 第二次买入（动用第一笔的收益）
sell2 = max(sell2, buy2 + p)    # 第二次卖出
```

**为什么「最多两笔」会自动成立**：`buy2` 的初值是负无穷，只有第一笔卖出后 `sell1` 变
非负才可能被激活；若第一笔不划算，`buy2` / `sell2` 就一直不触发，等价于只做一笔；四态
都可以保持不动，所以一笔不做也允许。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock_iii.py` / `.cpp`）：

```python
def max_profit_iii(prices):
    neg = float("-inf")
    buy1 = buy2 = neg
    sell1 = sell2 = 0
    for p in prices:
        buy1 = max(buy1, -p)
        sell1 = max(sell1, buy1 + p)
        buy2 = max(buy2, sell1 - p)
        sell2 = max(sell2, buy2 + p)
    return sell2
```

```cpp
int maxProfitIII(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long buy1 = NEG, sell1 = 0, buy2 = NEG, sell2 = 0;
    for (int p : prices) {
        buy1 = std::max(buy1, -static_cast<long long>(p));
        sell1 = std::max(sell1, buy1 + p);
        buy2 = std::max(buy2, sell1 - p);
        sell2 = std::max(sell2, buy2 + p);
    }
    return static_cast<int>(sell2);
}
```

- **复杂度**：时间 O(n)，空间 O(1)（状态数恒为 4）。
- **易错点**：四行松弛的顺序不能乱——`sell1` 要用更新后的 `buy1`、`buy2` 要用更新后的
  `sell1`，写反就漏掉「今天买今天卖」的情形；`sell1` / `sell2` 初值为 0（可以一笔不做）。
- **相似题**：188（把两笔推广成 k 笔，见下）；121（k=1）；122（k 不限）；309 / 714
  （状态机骨架上再各加一个约束，见下）。

### 188. 买卖股票的最佳时机 IV（困难）

**题目**：给定价格数组和整数 `k`，最多完成 `k` 笔交易，求最大利润。

**思路**：

把 123 的四个状态推广成 `2k` 个。用两个长度为 `k+1` 的数组维护每个阶段的最优收益：

- `buy[j]`：已完成 `j-1` 笔、且第 `j` 笔已买入时的最大收益；
- `sell[j]`：已完成 `j` 笔时的最大收益。

每读到一个价格 `p`，按 `j` 从小到大滚动更新这一天的所有阶段：

```
buy[j]  = max(buy[j],  sell[j-1] - p)   # 第 j 笔买入，接在第 j-1 笔收益之后
sell[j] = max(sell[j], buy[j] + p)      # 第 j 笔卖出
```

边界靠 `sell[0] = 0`（一笔都不做）自然兜住：`j = 1` 时 `buy[1] = max(buy[1], -p)` 正是
第一笔买入。因为 `buy[j]` 只用到 `sell[j-1]`、`sell[j]` 只用到 `buy[j]`，从左到右滚动即可，
同一天先卖再买也合法。答案是 `sell[k]`。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock_iv.py` / `.cpp`）：

```python
def max_profit_iv(k, prices):
    if k <= 0 or not prices:
        return 0
    neg = float("-inf")
    buy = [neg] * (k + 1)
    sell = [0] * (k + 1)
    for p in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return sell[k]
```

```cpp
int maxProfitIV(int k, const std::vector<int> &prices) {
    if (k <= 0 || prices.empty()) return 0;
    const long long NEG = -1e18;
    std::vector<long long> buy(k + 1, NEG), sell(k + 1, 0);
    for (int p : prices) {
        for (int j = 1; j <= k; ++j) {
            buy[j] = std::max(buy[j], sell[j - 1] - static_cast<long long>(p));
            sell[j] = std::max(sell[j], buy[j] + p);
        }
    }
    return static_cast<int>(sell[k]);
}
```

- **复杂度**：时间 O(n·k)，空间 O(k)。
- **易错点**：`k = 0` 直接返回 0；数组长度为 `k+1` 是为了容纳哨兵 `sell[0]`；`j` 必须
  从 1 开始（没有「第 0 笔交易」）；`buy` 初值为负无穷、`sell` 初值为 0。
- **相似题**：123（k=2 的展开版，可对照四个变量的写法）；121（k=1，但用更省的常数状态）；
  122（k 不限）；k 大于「天数的一半」时，本题也会退化成 122。

### 309. 买卖股票的最佳时机含冷冻期（中等）

**题目**：可以完成任意多笔交易，但**卖出股票的第二天不能买入**（有一天冷冻期），求最大利润。

**思路**：

122 只有「持仓 / 空仓」两态，是因为卖出后可以立刻再买。冷冻期把「卖出」拉成了一个必须
停留一天的中间态，于是设三个状态：

- `hold`：目前持有股票；
- `sold`：今天刚卖出（明天必须冷冻，不能买）；
- `rest`：空仓且已脱离冷冻期，随时可以买。

每读到价格 `p`，用**旧值**同时更新三态（避免互相污染）：

```
hold = max(hold, rest - p)   # 只有 rest 才允许买入
sold = hold + p              # 今天卖出，必然来自持仓
rest = max(rest, sold)       # 昨天的 sold 今天解冻，并入 rest
```

关键就一处：**买入的来源从 122 的 `cash` 换成了 `rest`**——`sold` 当天不能买，冷冻期便
表达清楚了。答案是空仓态的最大值 `max(rest, sold)`（最后一天卖出也是合法收尾）。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock_with_cooldown.py` / `.cpp`）：

```python
def max_profit_cooldown(prices):
    neg = float("-inf")
    hold, sold, rest = neg, neg, 0
    for p in prices:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    return max(rest, sold)
```

```cpp
int maxProfitCooldown(const std::vector<int> &prices) {
    const long long NEG = -1e18;
    long long hold = NEG, sold = NEG, rest = 0;
    for (int p : prices) {
        long long new_hold = std::max(hold, rest - p);
        long long new_sold = hold + p;
        long long new_rest = std::max(rest, sold);
        hold = new_hold;
        sold = new_sold;
        rest = new_rest;
    }
    return static_cast<int>(std::max(rest, sold));
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：三态必须**同时**用旧值更新；Python 用元组同时赋值，C++ 要用临时变量中转，
  否则 `rest` 会读到本天刚解冻的 `sold`、或 `sold` 读到本天新买的 `hold`；返回
  `max(rest, sold)` 而不是只看 `rest`。
- **相似题**：122（无冷冻期的对照）；714（把冷冻期换成手续费，见下）；买卖股票 I~IV
  （状态机骨架相同）。

### 714. 买卖股票的最佳时机含手续费（中等）

**题目**：可以完成任意多笔交易，但每笔交易（一次买入 + 一次卖出）需支付固定手续费
`fee`，求最大利润。

**思路**：

骨架与 122 完全一致，仍是 `hold` / `cash` 两态。唯一的区别是卖出时要扣手续费：

```
hold = max(hold, cash - p)
cash = max(cash, hold + p - fee)
```

手续费只在卖出时扣一次，天然对应「每笔完整交易收费一次」。因为收益已扣过费，价格涨幅
必须大于手续费才值得交易，DP 会自动在「继续持有」与「落袋付小费」之间取舍。

**代码**（`src/dynamic-programming/best_time_to_buy_and_sell_stock_with_fee.py` / `.cpp`）：

```python
def max_profit_fee(prices, fee):
    neg = float("-inf")
    hold, cash = neg, 0
    for p in prices:
        hold = max(hold, cash - p)
        cash = max(cash, hold + p - fee)
    return cash
```

```cpp
int maxProfitFee(const std::vector<int> &prices, int fee) {
    const long long NEG = -1e18;
    long long hold = NEG, cash = 0;
    for (int p : prices) {
        hold = std::max(hold, cash - static_cast<long long>(p));
        cash = std::max(cash, hold + p - fee);
    }
    return static_cast<int>(cash);
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：手续费只能扣一次，要放在**卖出**那一步（`hold + p - fee`），不能在买入
  和卖出各扣一次；`fee = 0` 时退化成 122；`hold` 初值负无穷。
- **相似题**：122（`fee = 0` 的特例）；309（用「冷冻」而非「收费」限制交易，见上）；
  188（次数受限的版本）。

---

## 综合：DP 的通用框架（五步流水线）

到这里，一维、序列、背包、网格、区间、状态机都走了一遍。回头看，33 道题其实是**同一条
流水线**在换不同的素材。拿到一道没见过的 DP 题，按下面五步依次追问，基本能落地：

1. **状态定义**：`dp[...]` 到底表示什么？先用一句自然语言说清，再决定它有几个下标。
   卡壳时优先想「题目求的东西能不能直接当状态」（如 118 的结果即状态表）。
2. **转移方程**：当前状态能从哪些更小的状态一步得到？把「一步操作」翻译成对状态的改动。
   数方案用加法，求最值用 `min`/`max`，判可达用 `or`——运算符就是题目的问法。
3. **初始化**：最小、无法再分的边界状态是什么值？注意「空状态」和「起点不止一个」两种
   情况（如 139 的空字符串、198 的两个起点）。边界的**语义**想清楚，比死背初值可靠。
4. **遍历顺序**：按什么顺序填表，才能保证用到的状态都已算好？一维通常从左到右；区间 DP
   必须按**区间长度**从小到大；背包则要区别「容量正序 / 倒序」。
5. **举例验证**：拿一个三四步就能手推的小样例，把 `dp` 表逐格填一遍，和答案对齐。这一步
   最容易被跳过，却是抓出转移写错、初值设错、越界访问的**唯一可靠手段**。

整篇里最容易翻车的，是第 1 步和第 5 步。下面用两道题把这两处各钉死一次：

- **152** 说明「一个状态装不下题目需要的信息」时，要主动**扩充状态**；
- **32** 说明转移写不出来时，要靠**手推样例**去定位「这一步到底和哪个状态有关」。

### 152. 乘积最大子数组（中等）

**题目**：给定整数数组 `nums`，找出乘积最大的连续子数组（至少一个元素），返回其乘积。

**思路**：

先照抄 53「最大子数组和」：设 `dp[i] = 以 nums[i] 结尾的最大乘积`，转移
`dp[i] = max(nums[i], dp[i-1] * nums[i])`。可是这样会错。比如 `[-2, 3, -4]`：到 3 时
`dp = 3`，到 -4 时要么 `-4`、要么 `3 * -4 = -12`，都会得到 3，但真正的答案是
`(-2) * 3 * (-4) = 24`。**漏掉的那个 `-12` 看似很小，一旦再乘一个负数就翻身了。**

问题的根源在状态定义：乘积的符号会翻转，**当前最小的负数可能转化成将来的最大正数**。
所以只留「最大值」这一个量是不够的，状态必须携带这个会互相转化的同伴。改成同时记录两端：

```
cur_max = 以当前元素结尾的最大乘积
cur_min = 以当前元素结尾的最小乘积（负得最厉害的那个，可能翻盘）
```

读到 `x` 时，以 `x` 结尾的乘积只可能来自三种起点：从 `x` 重新开始、接上旧的 `cur_max`、
接上旧的 `cur_min`。三者一起比较，大的交给 `cur_max`、小的交给 `cur_min`：

```
cur_max = max(x, cur_max * x, cur_min * x)
cur_min = min(x, cur_max * x, cur_min * x)
```

右边用的是**同一轮更新前的旧值**，所以要先算好三个候选再赋值。全局最优 `best` 一路上
取 `cur_max` 即可——答案同样不保证落在最后一格。

**代码**（`src/dynamic-programming/max_product_subarray.py` / `.cpp`）：

```python
def max_product(nums):
    cur_max = cur_min = best = nums[0]
    for x in nums[1:]:
        candidates = (x, cur_max * x, cur_min * x)
        cur_max = max(candidates)
        cur_min = min(candidates)
        best = max(best, cur_max)
    return best
```

```cpp
long long maxProduct(const std::vector<int> &nums) {
    long long curMax = nums[0], curMin = nums[0], best = nums[0];
    for (size_t i = 1; i < nums.size(); ++i) {
        long long x = nums[i];
        long long a = x, b = curMax * x, c = curMin * x;
        long long newMax = std::max({a, b, c});
        long long newMin = std::min({a, b, c});
        curMax = newMax;
        curMin = newMin;
        best = std::max(best, curMax);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`cur_max`、`cur_min` 必须**同时**用旧值更新；`min` 和 `max` 的候选集合
  完全相同，只差取哪个极值，漏掉 `cur_min * x` 就会重蹈上面的覆辙；C++ 用临时变量中转，
  不要先改 `curMax` 再拿它算 `curMin`；乘积可能越界，C++ 用 `long long` 承接。
- **相似题**：53（正数版本，只需保最大；这里的负数翻盘正是它省掉 min 的原因）；
  121 / 122（同样是「把会互相转化的量一起存」的状态机思路）。

### 32. 最长有效括号（困难）

**题目**：给定只含 `(` 和 `)` 的字符串 `s`，找出最长的、括号正确匹配的连续子串的长度。

**思路**：

要求「连续」，仍把状态钉在右端点：`dp[i] = 以 s[i] 结尾的最长有效括号长度`，显然
`s[i]` 必须是 `)`，否则 `dp[i] = 0`。难点全在转移，得看 `s[i]` 前面是谁：

- 若 `s[i-1] == '('`，即 `"...()"`，长度是 2，再拼上更早那段（以 `s[i-2]` 结尾的）：

  ```
  dp[i] = dp[i-2] + 2
  ```

- 否则 `s[i-1] == ')'`，说明 `s[i-1]` 处已有一段有效括号，长度为 `dp[i-1]`。这时能与
  `s[i]` 配对的那个 `(`，必须**正好落在这一整段的前面一格**：

  ```
  j = i - dp[i-1] - 1
  ```

  若 `j` 合法且 `s[j] == '('`，就把「中间整段」（`dp[i-1]`）、「这一对」（2）、以及
  `j` 之前那段（`dp[j-1]`）拼起来：

  ```
  dp[i] = dp[i-1] + 2 + dp[j-1]        # j >= 1 时
  ```

这里最能体会第 5 步「举例验证」的价值：**为什么配对位置必须跳过中间整段？** 拿
`"()(())"` 手推一遍——到最后一个 `)` 时，中间 `"(())"` 已经自成一段，外层的 `(` 只可能
在更靠前的位置，恰恰是 `j = i - dp[i-1] - 1` 算出来的那一格。光看公式容易写错，推两个
例子立刻清楚。答案同样不是 `dp[n-1]`，而是所有 `dp[i]` 的最大值。

本题也可用栈、或用左右两遍计数来做，这里只详展最能体现推理的 DP 版本。

**代码**（`src/dynamic-programming/longest_valid_parentheses.py` / `.cpp`）：

```python
def longest_valid_parentheses(s):
    n = len(s)
    dp = [0] * n
    best = 0
    for i in range(1, n):
        if s[i] == ')':
            if s[i - 1] == '(':
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2
            elif dp[i - 1] > 0:
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == '(':
                    dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
            best = max(best, dp[i])
    return best
```

```cpp
int longestValidParentheses(const std::string &s) {
    int n = s.size();
    std::vector<int> dp(n, 0);
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (s[i] == ')') {
            if (s[i - 1] == '(') {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (dp[i - 1] > 0) {
                int j = i - dp[i - 1] - 1;
                if (j >= 0 && s[j] == '(') {
                    dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
                }
            }
            best = std::max(best, dp[i]);
        }
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：配对位置是 `j = i - dp[i-1] - 1`，不是 `i - 1`，漏掉「中间整段」会少算；
  `j`、`j-1`、`i-2` 都要做越界保护；`s[i-1] == ')'` 时若 `dp[i-1] == 0` 直接跳过，别去
  访问越界的 `j`；结尾是 `(` 时 `dp[i]` 保持 0；答案取全局最大而非最后一格。
- **相似题**：20 / 22（括号匹配与生成，用栈或回溯）；139（切割型 DP，同样「钉前缀末尾 +
  枚举最后一段」）；53 / 152（「以 i 结尾」并在末尾取全局最优的同一骨架）。

---

## 规律总结

1. **DP 四步走：定义状态 → 写转移 → 定初值 → 排顺序。** 拿到一题先别写代码，先用一句话说清
   `dp[i]` 代表什么；状态定义对了，转移往往水到渠成。斐波那契、爬楼梯、打家劫舍共用同一个
   骨架，区别只在「状态是什么」和「怎么聚合前驱」。

2. **转移里的运算符，就是题目在问的东西。** 数方案用加法（70）、求最小值用 `min`（746）、
   求最大值用 `max`（198）；前驱集合相同、聚合方式不同，就得到不同的题。把 70 和 746 并排
   看，这条规律一目了然。

3. **初始化要覆盖「起点可以有好几个」和「空状态」。** 爬楼梯的地面算空走法（`dp[0]=1`）、
   打家劫舍可以任选起点（两个 0 起步）、最小花费有两个可达初值。初值错一格，整条递推就全错，
   务必想清楚每个边界状态的**语义**再赋值。

4. **只依赖前几项时，用滚动变量把空间压到 O(1)。** 斐波那契/爬楼梯/最小花费只依赖前两项、
   打家劫舍只依赖前两项，都能用两三个变量滚动。Python 用**元组同时赋值**、C++ 用临时变量
   中转，顺序写反就会互相覆盖。

5. **答案不一定落在最后一个状态。** 最小花费爬楼梯要取 `min(prev2, prev1)`，因为可以从倒数
   第二阶一步跨到顶。填完表后回头确认「题目要的终点是哪个状态」，别默认就是最后一个。

6. **二维 DP 只是「一维状态的笛卡尔积」。** 杨辉三角的 `dp[i][j]`、不同路径的网格，都是把
   一维下标扩成两个；转移从「上一格」变成「上方/左方」。先在一维上把四步走练熟，升维只是
   多画几个箭头。

7. **环状问题，剪一刀展成线。** 首尾相连时，先想「合法方案不可能同时满足什么」，然后在那个
   位置剪开，枚举少数几种「谁缺席」的情形分别做线性 DP。213 剪在首尾之间只需跑两遍；更复杂
   的环（如环形子数组最大和、下一步更大元素）也遵循同一个「化环为线 + 分类讨论」的思路。

8. **「连续」就把状态钉在结尾。** 连续子数组 / 子串的候选由右端点唯一确定，于是定义
   `dp[i] = 以第 i 个元素结尾的答案`；枚举所有结尾即覆盖所有候选。53（和）、674（长度）、
   718（公共后缀）是同一个骨架。**这类题的答案通常不是最后一个状态**，要再用全局最优变量
   收集，因为「以 i 结尾」只保证局部最优。

9. **「可不连续」就枚举前驱或升维。** 去掉连续性后，单序列的前驱变成「前面任意满足条件的
   元素」，于是 300 要 O(n²) 枚举；双序列则用 `dp[i][j]` 记录两个前缀的答案（1143/718/72）。
   判断该用哪种：**一个序列看结尾、两个序列看对角和邻格**。

10. **双序列 DP，先想「两个前缀的最后一个字符怎么配」。** 相等时通常零代价继承左上角；
    不等时看两个方向：跳过第一个串的末尾（左）、跳过第二个串的末尾（上）。1143 取
    `max`，72 取带代价的 `min`。把操作翻译成方向，是这类题的通用钥匙。

11. **边界就是「某一维为空」的语义。** 双序列的 `dp[0][j]`、`dp[i][0]` 往往不是 0，而是
    「从空串长出 j 个字符要几步」（72 里是 `j`）；三思边界含义，比死记初值可靠。

12. **滚动数组能省空间，但要先看清依赖方向。** 只依赖「上一行 + 本行左边」时，可以只留一
    行；此时若再依赖左上角，就必须在覆盖前先把旧值存下来。连续型的 718 滚动时更要小心从右
    往左更新 `j`，否则会用到本行刚被改写的数据。

13. **完全背包的标记是「容量正序」，0/1 背包的标记是「容量倒序」。** 正序时 `dp[i - w]`
    可能已被本轮更新，等于允许同一件物品再取一次；倒序时用到的都是旧值，每件至多取一次。
    322/279 用正序，416/494/1049 用倒序——只差一个字，语义天差地别。

14. **背包的「物品在外、容量在内」还是反过来，决定组合还是排列。** 物品外层：一个组合中
    物品顺序被固定，得到组合数（518）；容量外层：不同顺序算不同方案，得到排列数（377）。
    求「组合」时务必让物品在外层。

15. **同一张背包表能回答三类问题，靠运算符切换。** 问「最少几件/最大价值」用 `min`/`max`
    （322、279）；问「有几种凑法」用 `+=`（518、494）；问「能不能凑出」用 `or`（416、
    1049）。状态和循环骨架不变，**运算符就是问题的问法**。

16. **「划分、带符号、粉碎」多半能化成「凑一半」的子集和问题。** 416 把等和划分化成
    凑 `total/2`；494 把 `+/-` 化成凑 `(total+target)/2`；1049 把两两相消化成凑最接近
    `total/2`。遇到这类题先做代数变形，把它翻译成背包，再套模板。

17. **切割型 DP：状态钉在前缀末尾，枚举最后一个切点。** `dp[i]` 表示「前 `i` 个字符能否
    被覆盖」，转移时枚举 `j` 并检查尾段 `s[j:i]`（139 查字典、91 查解码、132 查回文）。
    它与「以 i 结尾」的连续型 DP 同源，区别只在判定条件。

18. **网格 DP 就是「二维状态 + 上方/左方聚合」。** 舞台是 `m×n` 网格、每步只向右/向下时，
    定义 `dp[i][j]` 为「走到 `(i, j)` 的答案」，转移从 `dp[i-1][j]`、`dp[i][j-1]` 聚合，
    需要时再加左上的 `dp[i-1][j-1]`。62/64 只差一个运算符，221 多看一眼左上角；**障碍物
    就是置 0 的洞**（63）。第一行/第一列通常要单独初始化，因为那里前驱唯一。

19. **区间 DP：把「顺序」反转成「最后一步」，依赖就断了。** 遇到「戳破 / 合并 / 删除一个
    元素会让左右合并」的题，正向依赖被破坏，改设 `dp[i][j]` 为区间上的最优解，**枚举最后
    处理的元素 `k`**，伤害/收益只与边界和 `k` 有关，左右两段各成子问题（312、1039、1000）。
    按**区间长度从小到大**填表，`dp` 常取开区间 `(i, j)` 配两端哨兵。

20. **回文类：看两端配不配，结论只有两种情况。** `dp[i][j]` 表示区间是否/多长回文，相等
    则缩进一格（子序列 `+2`、子串判真），不等则丢一边取 `max`（子序列）或直接判假（子串）。
    `i` 从大到小、`j` 从小到大，`j - i < 2` 用来兜住空串 / 单字符的 base case。516 和
    647 是同一张表的「求长度」与「数个数」两个问法。

21. **当状态是「手的姿态」时，用状态机 DP。** 买卖股票系列里，`dp` 记录的不是「第 i 天
    的答案」，而是「以某种姿态结束第 i 天时的最大收益」。状态少则两个（`hold` / `cash`，
    121/122/714），多则按交易笔数摊开（123 四个、188 有 `2k` 个），或再加一个中间态
    （309 的 `sold`）。**每天对所有状态做一轮松弛**，是这类题的统一动作。

22. **一天内先卖后买是允许的，所以多状态可以同时更新。** `hold` 与 `cash` 的松弛顺序
    （先更新谁）在无冷冻期时不影响答案；但一旦状态之间存在「当天不能互相转化」的约束
    （309 的 `sold` 当天不能变 `hold`），就必须**用旧值同时更新**——Python 靠元组同时
    赋值、C++ 靠临时变量中转。

23. **同族题的差别，几乎都落在「转移里的一个符号」。** 121→122 把买入的 `-p` 改成
    `cash - p`（一个交易变无限笔）；122→714 在卖出加 `-fee`；123→188 把写死的四个状态
    换成 `2k` 个循环变量；122→309 把买入来源 `cash` 换成 `rest` 并多一个中间态。
    **读这类题不要从头推，先问「它比最简骨架多改了什么」。**

24. **通用流水线：状态定义 → 转移 → 初始化 → 遍历顺序 → 举例验证，五步一步都不能省。**
    前四步决定代码怎么写，第五步决定它对不对。拿到新题先按顺序追问一遍；如果第 2 步写不出
    转移，八成要退回第 1 步重新定义状态，而不是硬凑公式。152 和 32 就是「验证」逼着我们把
    状态和转移想透的两个例子。

25. **一个状态装不下题目需要的信息，就主动扩充状态。** 152 的乘积会因为负数翻号，所以要把
    「最大」和「最小」两个伴随量一起存；309 要加「刚卖出」的中间态；股票系列按交易笔数把
    状态摊开。**判据是「将来会互相转化的量，现在就得都留着」**，别让当前的最优把潜力股丢掉。

26. **转移卡壳时，回去手推一个三四步的小样例。** 把 `dp` 表逐格填出来，再对照答案，往往一眼
    看出「这一步其实和哪个状态有关」。32 里配对位置要跳过中间整段，就是手推 `"()(())"` 才
    想明白的。**DP 的调试不靠猜，靠把表填一遍。**
