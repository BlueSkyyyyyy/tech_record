# 分治：拆开、解决、合并

分治（divide and conquer）不是某一道题的技巧，而是一种**把大问题拆成同类小问题**的
通用思路，只有三步：

1. **分解（divide）**：把原问题切成若干规模更小的同类子问题；
2. **解决（conquer）**：递归求解子问题，规模小到不能再分时直接给出答案（`base case`）；
3. **合并（combine）**：把子问题的解拼成原问题的解。

它之所以有效，是因为「切分」让问题的规模指数级缩小，而「合并」保证小问题的答案
能还原出大问题的答案。只要这两点成立，一段平凡的递归就能换成一份高效的算法。

本篇先用两道基础题立起两种最典型的合并方式，等后续补充分治篇时，再把归并排序、
翻转对、搜索二维矩阵等题目接上来。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：合并时从左右答案里挑一个 | 169. 多数元素 | 简单 |
| 模式二：合并时还要算「跨越分界」的答案 | 53. 最大子数组和 | 中等 |

两题都用分治求解，也都各有更快的解法（投票法、动态规划）。**学分治的重点不是背这两道题，
而是看清「分解 → 解决 → 合并」这三步在一段代码里长什么样**，以及什么时候该在合并层多花功夫。

---

## 模式一：合并时从左右答案里挑一个

**适用信号**：问题可以一分为二地递归，且「整体答案」能由「左半答案 + 右半答案」
比较、挑选得到，不需要扫描两侧的全部元素。

**核心动作**：递归函数返回一个**候选答案**；合并时先看左右候选是否相同，相同则直接采用；
不同则各自统计一下在整段的票数，多数者胜。

### 169. 多数元素（简单）

**题目**：给定一个大小为 `n` 的数组 `nums`，返回其中的多数元素。多数元素是指在数组中
出现次数大于 `n/2` 的元素。你可以假设数组非空，且给定的数组总是存在多数元素。

**思路**：

把数组从中间切成左右两半，分别递归求出两半的「多数候选」。合并分两种情况：

- 左右候选**相同**：那它显然也是整段的多数，直接返回；
- 左右候选**不同**：说明某一半的多数在另一半不占多数，于是数一数这两个候选在
  整段 `[lo, hi)` 里各出现多少次，出现更多的留下。

**为什么只需比较左右两个候选**：多数元素的定义是「出现次数 > n/2」。把它切成两半后，
它不可能在两半里都不超过一半——那样总次数最多只有 `n/2`，与定义矛盾。所以整段的多数
一定至少是某一半的多数，也就一定出现在「左半候选、右半候选」这两个值里。既然答案必在
其中，我们只要在这两个值里比票数就够了，根本不必检查其它元素。

**为什么用分治而不是哈希计数**：哈希一遍统计是 O(n) 的、更简单实用；这里选分治是为了
示范「合并时比较两个候选」这种最朴素的分治形态——不依赖额外空间，递归结构一眼可见。
面试里若被要求「不用额外空间」，投票法（Boyer-Moore）才是 O(n)、O(1) 的正解。

**代码**（完整可运行版见 `src/divide-conquer/majority_element.py` / `.cpp`）：

```python
def majority_element(nums):
    def count_in_range(lo, hi, target):
        count = 0
        for i in range(lo, hi):
            if nums[i] == target:
                count += 1
        return count

    def majority(lo, hi):
        if hi - lo == 1:
            return nums[lo]
        mid = (lo + hi) // 2
        left = majority(lo, mid)
        right = majority(mid, hi)
        if left == right:
            return left
        left_count = count_in_range(lo, hi, left)
        right_count = count_in_range(lo, hi, right)
        return left if left_count > right_count else right

    return majority(0, len(nums))
```

```cpp
int countInRange(const std::vector<int> &nums, int lo, int hi, int target) {
    int count = 0;
    for (int i = lo; i < hi; ++i) {
        if (nums[i] == target) ++count;
    }
    return count;
}

int majority(const std::vector<int> &nums, int lo, int hi) {
    if (hi - lo == 1) return nums[lo];
    int mid = (lo + hi) / 2;
    int left = majority(nums, lo, mid);
    int right = majority(nums, mid, hi);
    if (left == right) return left;
    int leftCount = countInRange(nums, lo, hi, left);
    int rightCount = countInRange(nums, lo, hi, right);
    return leftCount > rightCount ? left : right;
}

int majorityElement(const std::vector<int> &nums) {
    return majority(nums, 0, static_cast<int>(nums.size()));
}
```

- **复杂度**：时间 O(n log n)（递归树每层合计扫描 O(n)，共 log n 层），空间 O(log n)（递归栈）。
- **易错点**：区间统一用**左闭右开** `[lo, hi)`，因此 `base case` 是 `hi - lo == 1`
  （只剩一个元素），别写成 `lo == hi` 或 `lo > hi`；只在左右候选**不同**时才需要统计票数，
  相同可以直接返回，漏掉这个优化不影响正确性但会白扫一遍；统计范围是整段 `[lo, hi)`，
  不是某一半。
- **相似题**：53. 最大子数组和（同为分治，但合并不是挑候选而是算跨越值，见下）；
  169 的投票法解（Boyer-Moore，O(n)/O(1)）和哈希解可一并对比记忆。

---

## 模式二：合并时还要算「跨越分界」的答案

**适用信号**：答案对应数组上的一段区间，递归切开后，除了「完全在左、完全在右」，
还有「横跨分界点」这第三类情况，必须在合并层单独计算。

**核心动作**：递归求出左右两半各自的最优解；再从分界点 `mid` 向两侧扩展，算出所有
**跨过分界点**的区间的答案；三者取最大。

### 53. 最大子数组和（中等）

**题目**：给你一个整数数组 `nums`，请你找出一个具有最大和的连续子数组（子数组最少
包含一个元素），返回其最大和。

**思路**：

把数组从中间一切为二。任意一段连续子数组和分界点的关系只有三种：

1. 完全落在左半段；
2. 完全落在右半段；
3. 横跨分界点（左半占一截、右半占一截）。

前两类递归求解。第三类里，跨过 `mid` 的子数组必然形如「从 `mid` 往左延伸的一段后缀 +
从 `mid` 往右延伸的一段前缀」。要让总和最大，左右两侧就各自取**从 `mid` 出发的最大和**：
从 `mid-1` 向左累加、从 `mid` 向右累加，各自记录累加过程中出现过的最大值。两段最大前缀
相加，就是跨越中点的最大和。最后三种情况取最大即为答案。

**为什么这样不重不漏**：任何连续子数组要么跨 `mid`、要么不跨；不跨的必然整体落在
左半或右半。三类穷尽了全部可能，且互不重叠，所以取三者最大就是全局最优。

**为什么另一类是「合并层干活」**：和 169 不同，这里不能只靠比较左右子答案——真正的
最优区间可能一半在左、一半在右，它的和根本不等于左右两个局部最优之和。所以合并层
必须承担「跨越区间的计算」。这正是归并类分治（归并排序、求逆序对/翻转对）共同的形状：
递归负责两侧，合并负责跨界的贡献。

**更优解法**：动态规划（Kadane 算法）用 `dp[i]` 表示「以 `i` 结尾的最大子数组和」，
一次遍历 O(n) 即可，见动态规划篇。分治解慢一些（O(n log n)），但它是「区间 + 分治」
的标准模板，值得掌握。

**代码**（`src/divide-conquer/maximum_subarray.py` / `.cpp`）：

```python
def max_subarray(nums):
    def solve(lo, hi):
        if hi - lo == 1:
            return nums[lo]
        mid = (lo + hi) // 2
        left_best = solve(lo, mid)
        right_best = solve(mid, hi)

        best = float("-inf")
        total = 0
        for i in range(mid - 1, lo - 1, -1):
            total += nums[i]
            best = max(best, total)
        left_cross = best

        best = float("-inf")
        total = 0
        for i in range(mid, hi):
            total += nums[i]
            best = max(best, total)
        right_cross = best

        return max(left_best, right_best, left_cross + right_cross)

    return solve(0, len(nums))
```

```cpp
int solve(const std::vector<int> &nums, int lo, int hi) {
    if (hi - lo == 1) return nums[lo];
    int mid = (lo + hi) / 2;
    int leftBest = solve(nums, lo, mid);
    int rightBest = solve(nums, mid, hi);

    int best = INT_MIN, total = 0;
    for (int i = mid - 1; i >= lo; --i) {
        total += nums[i];
        best = std::max(best, total);
    }
    int leftCross = best;

    best = INT_MIN, total = 0;
    for (int i = mid; i < hi; ++i) {
        total += nums[i];
        best = std::max(best, total);
    }
    int rightCross = best;

    return std::max({leftBest, rightBest, leftCross + rightCross});
}

int maxSubArray(const std::vector<int> &nums) {
    return solve(nums, 0, static_cast<int>(nums.size()));
}
```

- **复杂度**：时间 O(n log n)（`T(n) = 2T(n/2) + O(n)`，合并扫描 O(n)），空间 O(log n)（递归栈）。
- **易错点**：`base case` 仍是 `hi - lo == 1`，返回 `nums[lo]` 而不是 0——子数组
  **至少包含一个元素**，全负数时答案是最大的那个负数，初值不能用 0；左右扩展时
  `left_cross` 从 `mid - 1` 开始、`right_cross` 从 `mid` 开始，两侧恰好拼成跨越中点的
  那段；C++ 里初值用 `INT_MIN`（需 `<climits>`），断言负数数组可防「默认 0」的坑；
  C++ 用 `std::max({a, b, c})` 一次取三值最大，需 `<algorithm>`。
- **相似题**：169. 多数元素（同为分治，合并方式对照，见上）；912. 排序数组（归并排序，
  与本题同源）、493. 翻转对（归并时统计跨界对数，见后续分治篇）；
  53 的动态规划解与 152. 乘积最大子数组（第 13 篇动态规划）。

---

## 规律总结

1. **分治只需回答三个问题**：怎么分（通常是取中点）、什么是最小子问题（`base case`
   直接给答案）、怎么合并。把这三件事写清楚，代码自然就出来了。

2. **`base case` 要能独立作答**：169 剩一个元素时它就是自己的多数；53 剩一个元素时
   最大子数组就是它本身（注意不是 0）。`base case` 写错，整棵递归都跟着错。

3. **合并是分治的灵魂**，也是最见功力的地方：
   - 169 的合并只是「左右候选票数比较」；
   - 53 的合并要额外计算「跨越分界点」的贡献。
   拿到一题先问：**合并时是否需要利用「跨界」的信息？** 需要，就得像 53 那样在合并层扫描。

4. **区间一律用左闭右开 `[lo, hi)`**：`mid = (lo + hi) // 2`，左半 `[lo, mid)`、
   右半 `[mid, hi)`，`base case` 写 `hi - lo == 1`。这套约定和 C++/Python 的下标习惯
   一致，能避免 `+1/-1` 的边界错误。

5. **复杂度看递推式**：若 `T(n) = 2T(n/2) + O(n)`（每层合并扫一遍），由主定理得
   O(n log n)；若合并是 O(1)，则是 O(n)。分治的复杂度基本就由「分几路、合并多贵」决定。

6. **分治往往不是最优，但它是「可并行、可迁移」的模板**：169 有 O(n) 投票法、
   53 有 O(n) 动态规划，分治都慢一档。可它的递归结构清晰、天然可并行，而且
   「合并层算跨界贡献」这一骨架能直接迁移到归并排序、逆序对、翻转对、区间统计等
   一大类题目上——这正是后续分治篇要继续展开的主线。
