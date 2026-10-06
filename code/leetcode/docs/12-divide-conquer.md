# 分治：拆开、解决、合并

分治（divide and conquer）不是某一道题的技巧，而是一种**把大问题拆成同类小问题**的
通用思路，只有三步：

1. **分解（divide）**：把原问题切成若干规模更小的同类子问题；
2. **解决（conquer）**：递归求解子问题，规模小到不能再分时直接给出答案（`base case`）；
3. **合并（combine）**：把子问题的解拼成原问题的解。

它之所以有效，是因为「切分」让问题的规模指数级缩小，而「合并」保证小问题的答案
能还原出大问题的答案。只要这两点成立，一段平凡的递归就能换成一份高效的算法。

分治最迷人的地方在**合并层**：题目越难，合并时要处理的信息往往越多。本专题从这里
出发，先用两道小题立起两种最基本的合并方式，再一路走到归并排序家族、分治建树、
以及「按分隔符切分」的字符串分治。你会反复看到同一个骨架，只是「合并」这一格
换了内容。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：合并时从左右答案里挑一个 | 169. 多数元素 | 简单 |
| 模式二：合并时还要算「跨越分界」的答案 | 53. 最大子数组和 | 中等 |
| 模式三：归并排序与「合并时统计」 | 912. 排序数组 · 493. 翻转对 | 中等 / 困难 |
| 模式四：链表的归并与多路归并 | 148. 排序链表 · 23. 合并 K 个升序链表 | 中等 / 困难 |
| 模式五：分治求幂与分治建树 | 50. Pow(x, n) · 654. 最大二叉树 | 中等 |
| 模式六：找到「分隔符」，一刀切开 | 241. 为运算表达式设计优先级 · 395. 至少有 K 个重复字符的最长子串 | 中等 |

同一模式下的题目放在一起，可以先读第一道、再体会第二道多了什么。**学分治的重点不是
背题，而是看清「分解 → 解决 → 合并」这三步在一段代码里长什么样**，以及什么时候
该在合并层多花功夫。

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
  与本题同源）、493. 翻转对（归并时统计跨界对数，见下）；
  53 的动态规划解与 152. 乘积最大子数组（动态规划篇）。

---

## 模式三：归并排序与「合并时统计」

**适用信号**：问题本身不一定是排序，但需要统计「满足某种大小关系的元素对」，
且这种关系只在两个已经有序的半段之间才好线性扫描。

**核心动作**：先递归把左右半段**排好序**，在**合并之前**用单调指针统计跨界贡献，
最后再做常规的合并。统计和合并是两趟独立扫描，互不干扰。

### 912. 排序数组（中等）

**题目**：给你一个整数数组 `nums`，请你将该数组升序排列。

**思路**：

这是分治在排序上的直接落地：

1. **分解**：把数组从中间切成左右两半；
2. **解决**：递归地把左半、右半分别排好序；
3. **合并**：把两个「已经有序」的半段合并成一个有序数组。

合并用双指针：两半各拿一个指针指向开头，每次取较小的那个放进结果，谁被取走谁后移。
因为两半各自有序，一趟就能合并完。

**为什么不会漏元素**：合并时两半的元素总会被某个指针扫到，且每次只取两半当前最小者，
保证结果非降序；一边取完时，另一边剩下的直接整体接上即可。

**为什么归并排序值得单独写**：它的「分 + 合」是分治的教科书模板。很多题
（求逆序对、翻转对、区间统计）都是在这套骨架上「合的时候多做一点事」，下一题就是
最好的例子。

**代码**（`src/divide-conquer/sort_array.py` / `.cpp`）：

```python
def sort_array(nums):
    n = len(nums)
    tmp = [0] * n

    def merge_sort(lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        merge_sort(lo, mid)
        merge_sort(mid, hi)

        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if nums[i] <= nums[j]:
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
            k += 1
        while i < mid:
            tmp[k] = nums[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = nums[j]
            j += 1
            k += 1
        nums[lo:hi] = tmp[lo:hi]

    merge_sort(0, n)
    return nums
```

```cpp
void mergeSort(std::vector<int> &nums, std::vector<int> &tmp, int lo, int hi) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    mergeSort(nums, tmp, lo, mid);
    mergeSort(nums, tmp, mid, hi);

    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (nums[i] <= nums[j]) tmp[k++] = nums[i++];
        else tmp[k++] = nums[j++];
    }
    while (i < mid) tmp[k++] = nums[i++];
    while (j < hi) tmp[k++] = nums[j++];
    for (int t = lo; t < hi; ++t) nums[t] = tmp[t];
}

std::vector<int> sortArray(std::vector<int> nums) {
    std::vector<int> tmp(nums.size());
    mergeSort(nums, tmp, 0, static_cast<int>(nums.size()));
    return nums;
}
```

- **复杂度**：时间 O(n log n)（`T(n) = 2T(n/2) + O(n)`，每层合计 O(n)，共 log n 层），
  空间 O(n)（临时数组）。
- **易错点**：`base case` 用 `hi - lo <= 1`（空或单个元素都已有序），比 `== 1` 更省心；
  合并时要**先把结果写进临时数组、再整体复制回原数组**，直接在 `nums` 上原地覆盖会丢失
  尚未取走的数据；`nums[i] <= nums[j]` 里的等号保证排序**稳定**（值相等时左边的先进结果）；
  区间沿用左闭右开 `[lo, hi)`。
- **相似题**：493. 翻转对（在归并前多统计一次跨界对数，见下）；148. 排序链表
  （同一套归并思路搬到链表上，见模式四）；912 的快速排序版（快排的分区思想与分治的
  「分解」相通，但重点在划分而非合并）。

### 493. 翻转对（困难）

**题目**：给定一个数组 `nums`，如果 `i < j` 且 `nums[i] > 2 * nums[j]`，就称 `(i, j)`
为一个「重要翻转对」。返回重要翻转对的总数。

**思路**：

暴力枚举所有 `(i, j)` 是 O(n²)。用分治把数组一分为二，任意一对 `(i, j)` 按位置关系
只可能：

1. `i`、`j` 都在左半；
2. `i`、`j` 都在右半；
3. `i` 在左半、`j` 在右半（跨界）。

前两类递归统计；第三类在合并阶段单独统计。统计跨界对的关键性质是「左右两半都各自
有序」——这发生在两半分别递归排好、但还没合并的时候。于是对左半每个 `i`，用一个指针
`j` 从右半开头往右滑，滑到第一个让 `nums[i] <= 2 * nums[j]` 的位置停下；此时右半在
`j` 之前的元素都满足条件，贡献 `j - mid` 对。因为左半有序，`i` 增大时 `nums[i]` 非降、
条件更难满足，`j` 只会继续右移，不会回退，所以整个统计是线性的。

**为什么先统计再合并**：统计要求左右两半有序才能用单调指针；而合并会破坏「分半」的
边界，所以统计必须放在 `merge` 之前。

**为什么用长整型比较**：`2 * nums[j]` 可能溢出 32 位 `int`，所以比较前先转成 64 位
再做乘法（Python 整数无溢出，C++ 用 `2LL` 提升）。

**代码**（`src/divide-conquer/reverse_pairs.py` / `.cpp`）：

```python
def reverse_pairs(nums):
    n = len(nums)
    tmp = [0] * n

    def merge_sort(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        count = merge_sort(lo, mid) + merge_sort(mid, hi)

        j = mid
        for i in range(lo, mid):
            while j < hi and nums[i] > 2 * nums[j]:
                j += 1
            count += j - mid

        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if nums[i] <= nums[j]:
                tmp[k] = nums[i]
                i += 1
            else:
                tmp[k] = nums[j]
                j += 1
            k += 1
        while i < mid:
            tmp[k] = nums[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = nums[j]
            j += 1
            k += 1
        nums[lo:hi] = tmp[lo:hi]
        return count

    return merge_sort(0, n)
```

```cpp
int mergeSort(std::vector<int> &nums, std::vector<int> &tmp, int lo, int hi) {
    if (hi - lo <= 1) return 0;
    int mid = (lo + hi) / 2;
    int count = mergeSort(nums, tmp, lo, mid) + mergeSort(nums, tmp, mid, hi);

    int j = mid;
    for (int i = lo; i < mid; ++i) {
        while (j < hi && static_cast<long long>(nums[i]) > 2LL * nums[j]) ++j;
        count += j - mid;
    }

    int i = lo, k = lo;
    j = mid;
    while (i < mid && j < hi) {
        if (nums[i] <= nums[j]) tmp[k++] = nums[i++];
        else tmp[k++] = nums[j++];
    }
    while (i < mid) tmp[k++] = nums[i++];
    while (j < hi) tmp[k++] = nums[j++];
    for (int t = lo; t < hi; ++t) nums[t] = tmp[t];
    return count;
}

int reversePairs(std::vector<int> nums) {
    std::vector<int> tmp(nums.size());
    return mergeSort(nums, tmp, 0, static_cast<int>(nums.size()));
}
```

- **复杂度**：时间 O(n log n)（递归 + 每层合并/统计各 O(n)），空间 O(n)（临时数组）。
- **易错点**：统计必须发生在 `merge` **之前**，且此时左右两半已经有序；指针 `j` 在
  统计后**不能重置**再用于合并——合并要重新从 `mid` 起步；判据是 `nums[i] > 2 * nums[j]`
  而不是 `nums[j] * 2 < nums[i]` 随手写反；C++ 必须用 `2LL`（或先转 `long long`）避免
  整数溢出；计数结果可能很大，C++ 版计数虽然按题面可回 `int`，稳妥起见可用 `long long`。
- **相似题**：912. 排序数组（同一套归并骨架，只是不多统计，见上）；剑指 Offer 51
  数组中的逆序对（统计条件换成 `nums[i] > nums[j]`，是本题的简化版）；315. 计算右侧
  小于当前元素的个数（用归并记录每个元素右侧比它小的个数，骨架同源）。

---

## 模式四：链表的归并与多路归并

**适用信号**：数据在链表上，排序或合并不能靠下标跳跃；或者要把多条有序序列并成一条。

**核心动作**：把「合并两个有序链表」当成唯一积木，配合**分治**把规模摊平。
链表上分治的「分解」靠快慢指针找中点断链，而不是取下标。

### 148. 排序链表（中等）

**题目**：给你链表的头结点 `head`，请将其按升序排列并返回排序后的链表。

**思路**：

数组的归并排序需要「切一刀」把区间分开，链表没有下标，但可以用快慢指针找到中点、
断开，得到两条子链；分别递归排序后，再用「合并两个有序链表」合并。整体还是三步：
找中点（分解）→ 递归排序（解决）→ 合并两条有序链（合并）。

**为什么链表适合归并而不是快排**：链表无法 O(1) 随机访问，快排的分区要来回跳跃，很别扭；
而归并只需顺序遍历，天然适配链表，还能做到「只改指针、不搬值」。

**为什么让慢指针停在中点的前一个位置**：这样可以用 `slow.next = None` 把链断成两半，
避免递归时互相纠缠。快指针从 `head.next` 起步，偶数长度时会取到偏左的中点。

**代码**（`src/divide-conquer/sort_list.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def sort_list(head):
    if head is None or head.next is None:
        return head

    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None

    left = sort_list(head)
    right = sort_list(mid)
    return merge_two(left, right)
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeTwo(ListNode *a, ListNode *b) {
    ListNode dummy;
    ListNode *tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

ListNode *sortList(ListNode *head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode *slow = head;
    ListNode *fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    ListNode *mid = slow->next;
    slow->next = nullptr;

    ListNode *left = sortList(head);
    ListNode *right = sortList(mid);
    return mergeTwo(left, right);
}
```

- **复杂度**：时间 O(n log n)（每层找中点 + 合并共 O(n)，共 log n 层），
  空间 O(log n)（递归栈；指针原地调整，不额外分配结点）。
- **易错点**：快速指针从 `head.next` 起步，配合 `while (fast && fast.next)`，这样中点
  才能停在偏左处、断链不会断出空链；断链前先保存 `mid = slow.next`，再置
  `slow.next = None`，顺序反了就找不到右半条链；`base case` 是「空链或只有一个结点」；
  合并时 `tail.next = a if a else b`（C++ `a ? a : b`），别写成无条件 `a` 导致丢链。
- **相似题**：23. 合并 K 个升序链表（把「合并两个」推广到「合并 k 个」，见下）；
  206. 反转链表、21. 合并两个有序链表（链表篇的基础积木）；912. 排序数组（同一套
  归并排序的数组版，见模式三）。

### 23. 合并 K 个升序链表（困难）

**题目**：给定一个链表数组，每个链表都已经按升序排列。请将所有链表合并成一个升序
链表并返回。

**思路**：

如果先把第 1 条和第 2 条合并、结果再和第 3 条合并……那么第 1 条链会被反复扫描
`k` 次，最坏退化到 O(kn)。分治把 `k` 条链两两配对：第 1 与第 2 合、第 3 与第 4 合……
一轮下来链表条数减半，每轮合并总量都是 O(n)，共 `log k` 轮，总复杂度 O(n log k)。

**为什么配对合并更均衡**：每条链在每一轮最多参与一次合并，随着轮数增加，每条链被扫描
的次数是 `log k` 左右，而不是被「一条龙」串起来时的 `k` 次。这和归并排序「每层都两两
合并」的均衡思想完全一致。

**为什么用迭代而不是递归**：把链表数组不断折半成「待合并的两组」，迭代版只需一个
`while` 循环，把相邻两条合并后放回新数组，直到只剩一条。

**另一种做法**：小顶堆多路归并（见堆篇），复杂度同为 O(n log k)，但要额外堆空间；
分治解只靠「合并两个有序链表」这一个积木。

**代码**（`src/divide-conquer/merge_k_sorted_lists.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two(a, b):
    dummy = ListNode()
    tail = dummy
    while a and b:
        if a.val <= b.val:
            tail.next = a
            a = a.next
        else:
            tail.next = b
            b = b.next
        tail = tail.next
    tail.next = a if a else b
    return dummy.next


def merge_k_lists(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged.append(merge_two(lists[i], lists[i + 1]))
            else:
                merged.append(lists[i])
        lists = merged
    return lists[0]
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x = 0, ListNode *n = nullptr) : val(x), next(n) {}
};

ListNode *mergeTwo(ListNode *a, ListNode *b) {
    ListNode dummy;
    ListNode *tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) {
            tail->next = a;
            a = a->next;
        } else {
            tail->next = b;
            b = b->next;
        }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

ListNode *mergeKLists(std::vector<ListNode *> lists) {
    if (lists.empty()) return nullptr;
    while (lists.size() > 1) {
        std::vector<ListNode *> merged;
        for (size_t i = 0; i < lists.size(); i += 2) {
            if (i + 1 < lists.size()) merged.push_back(mergeTwo(lists[i], lists[i + 1]));
            else merged.push_back(lists[i]);
        }
        lists = merged;
    }
    return lists[0];
}
```

- **复杂度**：时间 O(n log k)（`n` 为结点总数，`k` 为链表条数），空间 O(k)（迭代版存放
  中间结果）。
- **易错点**：空数组要单独返回 `None`/`nullptr`，否则最后访问 `lists[0]` 越界；奇数条时
  最后一条没有搭档，要原样放进下一轮；每轮用步长 2 遍历，别漏掉落单的一条；合并两个
  链表的 `tail.next = a if a else b` 收尾不能忘。
- **相似题**：148. 排序链表（先分治排序再合并，见上）；堆篇的 23（小顶堆多路归并，
  两种解法对照记忆）；21. 合并两个有序链表（本题的最小积木）。

---

## 模式五：分治求幂与分治建树

**适用信号**：问题每次能把规模「减半」且余下一小部分，或者能由「根 + 左右两个同类子问题」
唯一确定。

**核心动作**：把「减半」写成递归；建树类问题则是「本次选出一个根，其余元素按位置
自然分成左右两组」，分别递归。

### 50. Pow(x, n)（中等）

**题目**：实现 `pow(x, n)`，计算 `x` 的 `n` 次幂（`n` 为整数，可能为负）。

**思路**：

朴素连乘要做 `n` 次乘法，`n` 很大时慢。注意：

- `x^n = (x^(n/2))^2`（`n` 为偶数）
- `x^n = (x^(n/2))^2 * x`（`n` 为奇数）

只要算出「一半指数」的幂，再平方一次（奇数补乘一个 `x`），就能得到整个幂。指数每层
递归减半，只需 O(log n) 次乘法。

**为什么负指数能一并处理**：`x^(-n) = (1/x)^n`，先把底数取倒数、指数取正，之后走同一套
逻辑。

**为什么叫「快速幂」**：它把「乘 n 次」压成「乘 log n 次」，是分治在数值计算里最经典的
应用；同样思路可推广到矩阵快速幂、模意义下的快速幂。

**代码**（`src/divide-conquer/pow_x_n.py` / `.cpp`）：

```python
def my_pow(x, n):
    if n < 0:
        x = 1.0 / x
        n = -n

    def power(base, exp):
        if exp == 0:
            return 1.0
        half = power(base, exp // 2)
        if exp % 2 == 0:
            return half * half
        return half * half * base

    return power(x, n)
```

```cpp
double power(double base, long long exp) {
    if (exp == 0) return 1.0;
    double half = power(base, exp / 2);
    if (exp % 2 == 0) return half * half;
    return half * half * base;
}

double myPow(double x, long long n) {
    if (n < 0) {
        x = 1.0 / x;
        n = -n;
    }
    return power(x, n);
}
```

- **复杂度**：时间 O(log n)（指数每层减半），空间 O(log n)（递归栈）。
- **易错点**：`base case` 是 `exp == 0` 返回 1.0（含 `x^0 = 1`，包括 `0^0` 按题约定为 1）；
  负指数要先取倒数再转正；C++ 里取负指数时用 `long long` 承接，避免 `-INT_MIN` 溢出；
  不要写成 `power(base, exp/2) * power(base, exp/2)`，那会重复递归、退化成 O(n)。
- **相似题**：69. x 的平方根（二分答案，与快速幂对照「log 级」的不同来源，二分篇）；
  372. 超级次方、矩阵快速幂（同一思想在取模与矩阵上的推广）。

### 654. 最大二叉树（中等）

**题目**：给定一个不重复的整数数组 `nums`，构造最大二叉树：根是 `nums` 中的最大元素；
左子树由最大值左边那段递归构造，右子树由最大值右边那段递归构造。返回根结点。

**思路**：

题目已经把分治三步写清楚了：在当前区间里找到最大值（分解），它的位置把区间分成左右
两段；对左右两段分别递归构造（解决）；把递归得到的左右子树挂到当前根上（合并）。区间
为空时返回空结点，这是最小子问题。

**为什么最大值一定在根、且左右互不干扰**：规则要求最大值作根，而左右两段在位置上天然
被最大值隔开、各自只含比它小的值，所以递归构造出的两棵子树能拼出唯一确定的树。

**为什么用下标区间而不是切片**：切片会复制数组、增加开销；用 `[lo, hi)` 的下标区间在原
数组上操作，语义更清晰。

**更优解法**：单调栈可以做到 O(n)——从右往左扫，用递减栈直接确定每个结点的父结点。
分治版是 O(n²)（最坏如严格递增数组），但结构最直观，适合理解「由序列递归建树」。

**代码**（`src/divide-conquer/construct_maximum_binary_tree.py` / `.cpp`）：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def construct_maximum_binary_tree(nums):
    if not nums:
        return None
    max_index = 0
    for i in range(1, len(nums)):
        if nums[i] > nums[max_index]:
            max_index = i
    node = TreeNode(nums[max_index])
    node.left = construct_maximum_binary_tree(nums[:max_index])
    node.right = construct_maximum_binary_tree(nums[max_index + 1:])
    return node
```

```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(x), left(l), right(r) {}
};

TreeNode *build(const std::vector<int> &nums, int lo, int hi) {
    if (lo >= hi) return nullptr;
    int maxIndex = lo;
    for (int i = lo + 1; i < hi; ++i) {
        if (nums[i] > nums[maxIndex]) maxIndex = i;
    }
    TreeNode *node = new TreeNode(nums[maxIndex]);
    node->left = build(nums, lo, maxIndex);
    node->right = build(nums, maxIndex + 1, hi);
    return node;
}

TreeNode *constructMaximumBinaryTree(const std::vector<int> &nums) {
    return build(nums, 0, static_cast<int>(nums.size()));
}
```

- **复杂度**：时间 O(n²)（最坏如严格递增数组，每层只缩小一个元素；平均 O(n log n)），
  空间 O(n)（递归栈，最坏退化成链表）。
- **易错点**：`base case` 是区间为空（`not nums` / `lo >= hi`）返回空结点；左右子区间
  要**跳过最大值本身**（`[lo, maxIndex)` 与 `[maxIndex+1, hi)`）；本题数组元素
  **不重复**，找最大值无需处理并列；切片写法里 `nums[max_index + 1:]` 的 `+1` 别漏。
- **相似题**：105. 从前序与中序构造二叉树、106. 从中序与后序构造二叉树、
  108. 将有序数组转换为二叉搜索树（都是「定根 + 分段递归」的建树模板，见二叉树篇）；
  654 的单调栈解（O(n) 优化）。

---

## 模式六：找到「分隔符」，一刀切开

**适用信号**：问题的最优解**不可能跨过**某类元素/位置。把这类「禁地」当作分隔符切断，
在每一段里独立递归，规模自然缩小。

**核心动作**：扫描当前范围，找出所有不合法的位置；用它们把范围切成若干子段；
对每段递归求最优，再取最大。若没有任何分隔符，整段就是答案。

### 241. 为运算表达式设计优先级（中等）

**题目**：给你一个由数字和运算符（`+`、`-`、`*`）组成的字符串 `expression`，按不同的
加括号方式，返回所有可能的运算结果（顺序不限，允许重复）。

**思路**：

任何一种加括号方式，最终都会归结为「某一次运算作为整式的最后一步」。所以枚举表达式里
的每一个运算符，把它当作最外层运算：以它为界，左边是一个子表达式、右边是一个子表达式，
分别递归求出它们所有可能的结果，再做一次这个运算符的组合，把结果收集起来。表达式里
没有运算符（就是一个数字）时，直接返回这个数。

**为什么能不重不漏**：每种加括号方式都有唯一的「最后执行的运算符」，按这个运算符分类
正好一一对应；递归到子表达式时同样枚举它自己的最后一步。这本质上是在枚举所有形态的
**表达式树**。

**为什么结果会重复**：不同的括号方式可能算出相同的值（例如 `2*3-4*5` 的两种方式都得到
`-10`），题目允许重复，不用去重。

**代码**（`src/divide-conquer/different_ways_to_add_parentheses.py` / `.cpp`）：

```python
def diff_ways_to_compute(expression):
    if expression.isdigit():
        return [int(expression)]

    results = []
    for i, ch in enumerate(expression):
        if ch in "+-*":
            left = diff_ways_to_compute(expression[:i])
            right = diff_ways_to_compute(expression[i + 1:])
            for a in left:
                for b in right:
                    if ch == "+":
                        results.append(a + b)
                    elif ch == "-":
                        results.append(a - b)
                    else:
                        results.append(a * b)
    return results
```

```cpp
std::vector<int> diffWaysToCompute(const std::string &expression) {
    std::vector<int> results;
    for (int i = 0; i < static_cast<int>(expression.size()); ++i) {
        char c = expression[i];
        if (c == '+' || c == '-' || c == '*') {
            std::vector<int> left = diffWaysToCompute(expression.substr(0, i));
            std::vector<int> right = diffWaysToCompute(expression.substr(i + 1));
            for (int a : left) {
                for (int b : right) {
                    if (c == '+') results.push_back(a + b);
                    else if (c == '-') results.push_back(a - b);
                    else results.push_back(a * b);
                }
            }
        }
    }
    if (results.empty()) results.push_back(std::stoi(expression));
    return results;
}
```

- **复杂度**：时间与卡特兰数同阶（`n` 个运算符有 `Catalan(n)` 种括号方式，无条件记忆化时
  会重复计算子表达式，最坏指数级；加记忆化可降为多项式），空间 O(n)（递归栈）。`n` 很小，
  直接分治足够。
- **易错点**：`base case` 判「整串是不是数字」（Python 用 `isdigit`，C++ 用「没有运算符时
  结果为空」回退 `std::stoi`），不能靠长度判断；切分点只取运算符，左右子串要**跳过运算符
  本身**（`expression[i+1:]`）；运算符的 `else` 分支只能留给 `*`，不要漏判 `-`。
- **相似题**：95. 不同的二叉搜索树 II（枚举根、左右递归生成所有形态，与本题枚举「最后
  一步运算符」同构，二叉树/分治交叉）；395. 至少有 K 个重复字符的最长子串（同一个
  「按分隔符切开再递归」的思路，见下）。

### 395. 至少有 K 个重复字符的最长子串（中等）

**题目**：给你一个字符串 `s` 和一个整数 `k`，找出 `s` 中的最长子串，要求该子串中的
每一个字符出现次数都不少于 `k`。返回该子串的长度。

**思路**：

如果某个字符在整个串里出现的次数都不足 `k`，那么任何满足条件的子串都**不可能包含它**
——子串里的次数只会更少。于是这个字符天然就是一道「分隔符」：合法的子串只可能落在它
切出的某一段里。把所有这类「非法字符」都当作分隔符，把串切成若干段，递归地在每段里
找最长合法子串，取最大即可。如果一段里所有字符的出现次数都不少于 `k`，整段就是合法的，
直接返回它的长度。

**为什么可以放心把非法字符扔掉**：合法子串对「每个含有的字符」都有下界要求，含了非法
字符就永远满足不了；所以最优解一定不含任何非法字符，去掉它们不会丢解，反而缩小了问题。

**为什么分段递归是自洽的**：每段内部的字符集合是原串的子集，用原串统计出来的「非法
字符」在段内依然非法（次数只会更少或不变），所以按同样规则继续切分不会错；递归到某段
没有非法字符时，就是该段的答案。

**代码**（`src/divide-conquer/longest_substring_with_at_least_k_repeating.py` / `.cpp`）：

```python
def longest_substring(s, k):
    if len(s) < k:
        return 0
    for ch in set(s):
        if s.count(ch) < k:
            return max(longest_substring(part, k) for part in s.split(ch))
    return len(s)
```

```cpp
int longestSubstring(const std::string &s, int k) {
    if (static_cast<int>(s.size()) < k) return 0;

    int cnt[26] = {0};
    for (char c : s) ++cnt[c - 'a'];

    char bad = 0;
    for (int i = 0; i < 26; ++i) {
        if (cnt[i] > 0 && cnt[i] < k) {
            bad = static_cast<char>('a' + i);
            break;
        }
    }
    if (bad == 0) return static_cast<int>(s.size());

    int best = 0, start = 0;
    for (int i = 0; i <= static_cast<int>(s.size()); ++i) {
        if (i == static_cast<int>(s.size()) || s[i] == bad) {
            if (i > start) best = std::max(best, longestSubstring(s.substr(start, i - start), k));
            start = i + 1;
        }
    }
    return best;
}
```

- **复杂度**：最坏 O(n²)（每层切分扫描 O(n)，最坏递归 O(n) 层，例如 `k` 很大时），
  空间 O(n)（递归栈 + 子串切片）。
- **易错点**：`base case` 只有一个——「当前串里所有字符频次都 ≥ k 就返回整段长度」，
  别把「长度 < k」当唯一出口；递归要真的**按非法字符切分**（`s.split(ch)` / 手动跳过），
  不能只对半切；统计频次针对当前串，每层重新统计；空片段要跳过（C++ 里 `i > start`）。
- **相似题**：241. 为运算表达式设计优先级（同为「找分隔符切开再递归」，见上）；
  76. 最小覆盖子串、438. 字母异位词（滑动窗口篇，与本题「字符计数约束」形成对照）；
  3. 无重复字符的最长子串（最长合法子串的另一个经典版本，滑动窗口篇）。

---

## 规律总结

1. **分治只需回答三个问题**：怎么分（取中点、取最大值、取运算符、取非法字符）、
   什么是最小子问题（`base case` 直接给答案）、怎么合并。把这三件事写清楚，代码自然
   就出来了。

2. **`base case` 要能独立作答，且口径要一致**：169 剩一个元素时它就是自己的多数；
   53 剩一个元素时最大子数组就是它本身（注意不是 0）；50 是指数归零返回 1；
   654 是区间为空返回空结点。`base case` 写错，整棵递归都跟着错。

3. **合并是分治的灵魂，也是最见功力的地方**：
   - 169 的合并只是「左右候选票数比较」；
   - 53 的合并要额外算「跨越分界点」的贡献；
   - 912/493 的合并是「先排序、再统计、最后拼接」；
   - 148/23 的合并就是「合并两条有序链」。
   拿到一题先问：**合并时是否需要利用「跨界」的信息？** 需要，就得在合并层多干活。

4. **区间一律用左闭右开 `[lo, hi)`**：`mid = (lo + hi) // 2`，左半 `[lo, mid)`、
   右半 `[mid, hi)`，`base case` 写 `hi - lo <= 1` 或 `hi - lo == 1`。这套约定和
   C++/Python 的下标习惯一致，能避免 `+1/-1` 的边界错误。链表没有下标，改用快慢指针
   找中点；此时注意让快指针从 `head.next` 起步、断链前先保存后半段。

5. **复杂度看递推式**：若 `T(n) = 2T(n/2) + O(n)`（每层合并扫一遍），由主定理得
   O(n log n)（912、493、53）；若合并是 O(1)，则是 O(n)；若每层只缩小一个元素，则退化
   成 O(n²)（654 最坏）。分治的复杂度基本由「分几路、合并多贵」决定。

6. **分治往往不是最优，但它是「可并行、可迁移」的模板**：169 有 O(n) 投票法、
   53 有 O(n) 动态规划、654 有 O(n) 单调栈、23 有 O(n log k) 的堆解。分治通常慢一档，
   可它的递归结构清晰、天然可并行，而且「合并层做文章」这一骨架能直接迁移到归并排序、
   逆序对、翻转对、区间统计等一大类题上。

7. **「按分隔符切分」是分治的另一副面孔**：241 用运算符切分、395 用频次不足的字符
   切分。它们的共同点是——找到问题里「最优解不可能跨过」的位置，用它把问题切成独立
   子问题。识别出这种「禁地」，往往就是难题的突破口。
