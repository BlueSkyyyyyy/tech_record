# 数组与双指针

数组题千变万化，但真正高频的「招式」只有几类。本篇先把最通用的两类双指针讲透：
**对撞双指针**（区间从两端向中间收缩）与**快慢指针**（一个探路、一个写结果）。
把这两类吃透，再看后面的滑动窗口、二分查找会顺很多。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 对撞双指针 | 167. 两数之和 II | 简单 |
| 对撞双指针 + 贪心 | 11. 盛最多水的容器 | 中等 |
| 对撞双指针 + 去重 | 15. 三数之和 | 中等 |
| 快慢指针（原地覆盖） | 26. 删除有序数组中的重复项 | 简单 |
| 快慢指针（原地交换） | 283. 移动零 | 简单 |

---

## 模式一：对撞双指针

**适用信号**：数组**有序**，或题目要求「两端向中间」逼近（两数之和、面积、回文判断）。

核心动作只有三步：判断当前 `lo`、`hi` 的组合是否满足条件；不满足时决定移动哪一端；每一步都要能**排除掉一整批不可能的解**。

### 167. 两数之和 II - 输入有序数组（简单）

**题目**：数组按下标从 1 开始、非递减排列。找出两个数，其和等于 `target`，返回这两个数的下标（1-indexed）。恰有一个答案，同一元素不能用两次。

**思路（为什么可以往中间收缩）**：
设 `lo` 在最左、`hi` 在最右，看 `a[lo] + a[hi]`：

- 等于 `target`：找到答案；
- 小于 `target`：需要更大。此时 `a[lo]` 与**任何** `a[j]`（`j < hi`）相加都 ≤ `a[lo] + a[hi] < target`，说明 `a[lo]` 不可能再有任何解，`lo` 可以安全右移；
- 大于 `target`：对称地，`hi` 可以安全左移。

每一步都排除一个元素，指针必然相遇，因此正确且是线性的。注意题目数组**已排序**，这是能一次扫描的前提；无序版本（两数之和 I）要用哈希表（见后续哈希篇）。

**代码**（完整可运行版见 `src/array/two_sum_ii.py` / `.cpp`）：

```python
def two_sum_sorted(numbers, target):
    lo, hi = 0, len(numbers) - 1
    while lo < hi:
        s = numbers[lo] + numbers[hi]
        if s == target:
            return [lo + 1, hi + 1]
        if s < target:
            lo += 1
        else:
            hi -= 1
    return []
```

```cpp
std::vector<int> twoSumSorted(const std::vector<int> &numbers, int target) {
    int lo = 0, hi = static_cast<int>(numbers.size()) - 1;
    while (lo < hi) {
        int sum = numbers[lo] + numbers[hi];
        if (sum == target) return {lo + 1, hi + 1};
        if (sum < target) ++lo;
        else --hi;
    }
    return {};
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：返回的是 **1-indexed**（要 `+1`）；循环条件是 `lo < hi` 而不是 `<=`，因为同一元素不能用两次。
- **相似题**：1. 两数之和（无序 → 哈希）；653. 两数之和 IV - BST；167 的变体「求两数之和最接近 target」。

### 11. 盛最多水的容器（中等）

**题目**：数组 `height` 表示每根竖线的高度。选两条线，与 x 轴围成的容器装水最多。面积 = `min(height[lo], height[hi]) * (hi - lo)`。

**思路（贪心双指针）**：
从最宽的两端开始（此时宽度最大）。每次移动**较矮的那一端**：

- 宽度只会减小，若想面积变大，只能指望高度变大；
- 高度由较短端决定，所以移动较高端没有意义——矮端不变、宽度还变小，面积只会更小。

于是「移动矮端」可以排除掉矮端当前作为答案的可能，且不遗漏更优解。这就是用**贪心**驱动指针移动的对撞双指针。

**代码**（`src/array/container_with_most_water.py` / `.cpp`）：

```python
def max_area(height):
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        h = min(height[lo], height[hi])
        best = max(best, h * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best
```

```cpp
int maxArea(const std::vector<int> &height) {
    int lo = 0, hi = static_cast<int>(height.size()) - 1, best = 0;
    while (lo < hi) {
        int h = std::min(height[lo], height[hi]);
        best = std::max(best, h * (hi - lo));
        if (height[lo] < height[hi]) ++lo;
        else --hi;
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：相等时移动哪一端都可以（两端都矮）；关键是不要移动高的一端。
- **相似题**：42. 接雨水（改成「能接多少水」，用单调栈或双指针）；167（同为对撞双指针，但移动规则不同）。

### 15. 三数之和（中等）

**题目**：返回所有和为 0 的三元组 `[a, b, c]`，要求三元组**不重复**。

**思路（排序 + 固定一个数 + 对撞双指针）**：
把三数之和降维成两数之和：

1. **排序**。排序有两个作用：让双指针可用；让重复元素相邻，方便去重。
2. 外层枚举第一个数 `nums[i]`，内层在 `(i, n-1]` 上用对撞双指针找「两数之和 = `-nums[i]`」，这正好是 167 的套路。
3. **去重**是本题真正的难点，要处理两处：
   - 外层：若 `nums[i] == nums[i-1]`，说明这个首数已用过，跳过；
   - 内层命中一组解后，`lo`、`hi` 要各自跳过相邻的相同值，否则同一组解会被重复计入。
4. **剪枝**：排序后若 `nums[i] > 0`，后面全是正数，和不可能为 0，直接 `break`。

**代码**（`src/array/three_sum.py` / `.cpp`）：

```python
def three_sum(nums):
    nums = sorted(nums)
    n = len(nums)
    res = []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
    return res
```

```cpp
std::vector<std::vector<int>> threeSum(std::vector<int> nums) {
    std::sort(nums.begin(), nums.end());
    int n = static_cast<int>(nums.size());
    std::vector<std::vector<int>> res;
    for (int i = 0; i < n - 2; ++i) {
        if (nums[i] > 0) break;
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        int lo = i + 1, hi = n - 1;
        while (lo < hi) {
            int s = nums[i] + nums[lo] + nums[hi];
            if (s < 0) ++lo;
            else if (s > 0) --hi;
            else {
                res.push_back({nums[i], nums[lo], nums[hi]});
                ++lo; --hi;
                while (lo < hi && nums[lo] == nums[lo - 1]) ++lo;
                while (lo < hi && nums[hi] == nums[hi + 1]) --hi;
            }
        }
    }
    return res;
}
```

- **复杂度**：排序 O(n log n)，主体 O(n²)；总时间 O(n²)，额外空间 O(1)（不计排序和输出）。
- **易错点**：忘记任一处去重都会超时或产生重复三元组；`break` 的剪枝条件是 `nums[i] > 0`。
- **相似题**：16. 最接近的三数之和；18. 四数之和（外层再套一层）；259. 较小的三数之和。

---

## 模式二：快慢指针（原地数组）

**适用信号**：要求**原地**修改数组，且需要「保留满足条件的元素、丢弃其余」。常见于去重、移除元素、移动零。

套路统一：`slow` 指向「下一个要写入的位置」，`fast` 负责扫描。`fast` 每找到一个需要保留的元素，就写到 `slow` 处并让 `slow` 前进。这样保留元素的**相对顺序不变**，且只遍历一遍。

### 26. 删除有序数组中的重复项（简单）

**题目**：非严格递增数组，原地去重，返回新长度 `k`；前 `k` 个位置为结果，后面是什么无所谓。

**思路**：数组有序 ⇒ 重复元素必然相邻。`slow` 指向已去重部分的最后一个位置，`fast` 从 1 开始扫；只要 `nums[fast] != nums[slow]`，说明遇到一个新值，写到 `slow+1`。

**代码**（`src/array/remove_duplicates_sorted.py` / `.cpp`）：

```python
def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

```cpp
int removeDuplicates(std::vector<int> &nums) {
    if (nums.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != nums[slow]) {
            ++slow;
            nums[slow] = nums[fast];
        }
    }
    return slow + 1;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：空数组要单独返回 0；返回的是 `slow + 1`（长度），不是下标。
- **相似题**：80. 删除有序数组中的重复项 II（每个元素最多留 2 个，比较 `nums[fast]` 与 `nums[slow-1]`）；27. 移除元素（把「不等于 val」作为保留条件）。

### 283. 移动零（简单）

**题目**：把所有 0 移到末尾，保持非零元素的相对顺序，必须原地。

**思路**：和 26 同构。把「非零」作为保留条件：`fast` 遇到非零就与 `slow` 交换，`slow` 前进。交换（而不是直接赋值）让 0 自然被换到后面，最后不必再补零。

**代码**（`src/array/move_zeroes.py` / `.cpp`）：

```python
def move_zeroes(nums):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1
```

```cpp
void moveZeroes(std::vector<int> &nums) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != 0) {
            std::swap(nums[slow], nums[fast]);
            ++slow;
        }
    }
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`slow <= fast` 恒成立，交换不会破坏已处理部分；若用赋值写法，记得最后把 `slow` 之后补 0。
- **相似题**：26、27、80（同一模板换保留条件）；905. 按奇偶排序数组。

---

## 规律总结

1. **对撞双指针 = 有序 + 两端逼近 + 可证明的排除**。`167` 排除的是「矮端那一侧」，`11` 排除的是「较矮的那根线」，`15` 先排序再降维成 `167`。判断移动哪一端，本质是问「当前这一端还有没有可能出现在最优解里」。
2. **快慢指针 = 保留条件 + 原地写入**。`slow` 是写指针，`fast` 是读指针。去重、移除、移动零都是「换一个保留条件」的同一道题。
3. **排序往往是双指针的前置步骤**：它同时带来「有序可收缩」和「相同元素相邻便于去重」两个好处。
4. 遇到「无序数组两数之和」不要硬套对撞双指针，那要用哈希表（下一篇）。
