# 数组与双指针

数组题千变万化，但真正高频的「招式」只有几类。本篇围绕双指针展开，讲透四种最常用的变形：
**对撞双指针**（区间从两端向中间收缩）、**快慢指针**（一个探路、一个写结果）、
**从后向前双指针**（原地归并时避免覆盖）与**原地反转技巧**（用三次反转完成轮转）。
把这几种吃透，再看后面的滑动窗口、二分查找会顺很多。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 对撞双指针 | 125. 验证回文串 | 简单 |
| 对撞双指针 | 167. 两数之和 II | 简单 |
| 对撞双指针 + 贪心 | 11. 盛最多水的容器 | 中等 |
| 对撞双指针 + 去重 | 15. 三数之和 | 中等 |
| 快慢指针（原地覆盖） | 26. 删除有序数组中的重复项 | 简单 |
| 快慢指针（原地覆盖） | 27. 移除元素 | 简单 |
| 快慢指针（回头看两位） | 80. 删除有序数组中的重复项 II | 中等 |
| 快慢指针（原地交换） | 283. 移动零 | 简单 |
| 从后向前双指针 | 88. 合并两个有序数组 | 简单 |
| 原地反转 | 189. 轮转数组 | 中等 |

---

## 模式一：对撞双指针

**适用信号**：数组**有序**，或题目要求「两端向中间」逼近（两数之和、面积、回文判断）。

核心动作只有三步：判断当前 `lo`、`hi` 的组合是否满足条件；不满足时决定移动哪一端；每一步都要能**排除掉一整批不可能的解**。

### 125. 验证回文串（简单）

**题目**：给定字符串 `s`，只考虑其中的字母和数字、忽略大小写，判断它是否为回文串。例如 `"A man, a plan, a canal: Panama"` 是回文，因为它化简后是 `"amanaplanacanalpanama"`。

**思路（跳过干扰字符的对撞双指针）**：
把 `lo`、`hi` 分别放在两端，每次先让它们各自挪到下一个**字母或数字**上（跳过空格和标点），再比较小写形式：

- 只要有一对字符不相等，立刻返回 `False`；
- 全部比对完（`lo` 追上了 `hi`）则返回 `True`。

为什么可以跳过非字母数字字符：回文只关心正读反读是否一致，而某一对字符是否相等只取决于这两个字符本身，与它们之间的空格标点无关。所以跳过干扰字符不会影响判断结果，一次线性扫描即可。这也是后续字符串篇里「回文判断」的通用骨架。

**代码**（完整可运行版见 `src/array/valid_palindrome.py` / `.cpp`）：

```python
def is_palindrome(s):
    lo, hi = 0, len(s) - 1
    while lo < hi:
        while lo < hi and not s[lo].isalnum():
            lo += 1
        while lo < hi and not s[hi].isalnum():
            hi -= 1
        if s[lo].lower() != s[hi].lower():
            return False
        lo += 1
        hi -= 1
    return True
```

```cpp
bool isPalindrome(const std::string &s) {
    int lo = 0, hi = static_cast<int>(s.size()) - 1;
    while (lo < hi) {
        while (lo < hi && !std::isalnum(static_cast<unsigned char>(s[lo]))) ++lo;
        while (lo < hi && !std::isalnum(static_cast<unsigned char>(s[hi]))) --hi;
        if (std::tolower(static_cast<unsigned char>(s[lo])) !=
            std::tolower(static_cast<unsigned char>(s[hi]))) {
            return false;
        }
        ++lo;
        --hi;
    }
    return true;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：内层两个 `while` 也要带上 `lo < hi` 守卫，否则全是标点的字符串里指针会越界；C++ 里 `isalnum`/`tolower` 要先转 `unsigned char`，否则负字符是未定义行为。
- **相似题**：167、11、15（本篇同为对撞双指针）；5. 最长回文子串、9. 回文数、680. 验证回文串 II（允许删一个字符）。

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
- **相似题**：80. 删除有序数组中的重复项 II（见下，每个元素最多留 2 个，比较 `nums[fast]` 与 `nums[slow-2]`）；27. 移除元素（见下，把「不等于 val」作为保留条件）。

### 27. 移除元素（简单）

**题目**：给定数组 `nums` 和值 `val`，原地移除所有等于 `val` 的元素，返回新长度 `k`。前 `k` 个位置保存结果，顺序不限，后面是什么无所谓。

**思路（换一个保留条件）**：
与 26 完全同构。`slow` 指向下一个要写入的位置，`fast` 扫描全数组：只要 `nums[fast] != val`，就把它写到 `slow` 处并让 `slow` 前进。
这里直接用**赋值**而非交换即可——被跳过的 `val` 会在后续写入时被覆盖；只有当你需要保留被丢弃元素的某种顺序时才要交换（见 283 移动零）。
换句话说，26、27、80 共用一套模板，区别只在「什么算需要保留」。

**代码**（`src/array/remove_element.py` / `.cpp`）：

```python
def remove_element(nums, val):
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

```cpp
int removeElement(std::vector<int> &nums, int val) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            ++slow;
        }
    }
    return slow;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：返回的是长度 `slow`；题目说顺序可以任意，所以不必纠结被删元素的去向；不要额外开数组。
- **相似题**：26（保留条件为「与前一个不同」）；80（保留条件为「与倒数第二个不同」）；283. 移动零（保留条件为「非零」，且需要交换）。

### 80. 删除有序数组中的重复项 II（中等）

**题目**：非严格递增数组，原地去重，让每个元素**最多出现两次**，返回新长度 `k`。

**思路（快慢指针 + 回头看两位）**：
`slow` 指向下一个要写入的位置，`fast` 负责扫描。判断 `nums[fast]` 能否保留，只需看它与「已写部分的倒数第二个元素」`nums[slow-2]` 是否相同：

- 若 `slow < 2`：还没写满两个，任何元素都能保留；
- 若 `nums[fast] == nums[slow-2]`：说明这个值已经出现过至少两次，再写就超标，跳过；
- 否则保留。

为什么只需比较 `nums[slow-2]`：数组有序，相同的值总是连续出现。若 `nums[fast]` 与倒数第二个相同，说明该值已连续写了两份，再写就是第三份；若不同，则要么是新值，要么上一份还没写满两份。推广开来，把「2」换成「m」，模板只需改为比较 `nums[slow-m]`。

**代码**（`src/array/remove_duplicates_sorted_ii.py` / `.cpp`）：

```python
def remove_duplicates_ii(nums):
    slow = 0
    for fast in range(len(nums)):
        if slow < 2 or nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

```cpp
int removeDuplicatesII(std::vector<int> &nums) {
    int slow = 0;
    for (int fast = 0; fast < static_cast<int>(nums.size()); ++fast) {
        if (slow < 2 || nums[fast] != nums[slow - 2]) {
            nums[slow] = nums[fast];
            ++slow;
        }
    }
    return slow;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`slow - 2` 越界由 `slow < 2` 短路保护，两个条件顺序不能反；`fast` 从 0 开始（因为可能整段都在覆盖自己）。
- **相似题**：26（最多留 1 个）；27；把常数 2 推广到 k 的「每个元素最多出现 k 次」模板题。

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

## 模式三：从后向前双指针（原地归并）

**适用信号**：两个**有序**序列要合并，且结果数组的**尾部是空的**（有足够空间）。典型就是 88。

从前往后写会覆盖 `nums1` 里尚未处理的小元素，所以要额外开数组。但从后往前写就不会：待写入位置永远在尚未读取的元素之后，天然安全。这个「哪边有空位就从哪边写」的思想在归并排序里反复出现。

### 88. 合并两个有序数组（简单）

**题目**：`nums1` 长度为 `m + n`，前 `m` 个元素有序；`nums2` 长度为 `n` 且有序。把 `nums2` 合并进 `nums1`，使 `nums1` 整体有序，原地完成。

**思路（三指针从后往前）**：
设 `i = m - 1` 指向 `nums1` 有效部分的末尾，`j = n - 1` 指向 `nums2` 的末尾，`k = m + n - 1` 指向 `nums1` 待写入的末尾。
每轮比较 `nums1[i]` 与 `nums2[j]`，把**较大**的放到 `nums1[k]`，然后相应指针左移。
为什么安全：`k` 始终不小于 `i`，写入位置不会盖住还没读的 `nums1` 元素；当 `nums2` 取完（`j < 0`）时，`nums1` 剩余部分本来就在正确位置，无需再动。循环条件只需判断 `j >= 0`。

**代码**（`src/array/merge_sorted_array.py` / `.cpp`）：

```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
```

```cpp
void merge(std::vector<int> &nums1, int m, std::vector<int> &nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (j >= 0) {
        if (i >= 0 && nums1[i] > nums2[j]) {
            nums1[k] = nums1[i];
            --i;
        } else {
            nums1[k] = nums2[j];
            --j;
        }
        --k;
    }
}
```

- **复杂度**：时间 O(m + n)，空间 O(1)。
- **易错点**：`i >= 0` 守卫不可少（`nums2` 还没取完时 `nums1` 可能已空）；循环只写 `j >= 0`；比较用 `>` 时相等取 `nums2` 的元素，不影响正确性。
- **相似题**：21. 合并两个有序链表（链表版，从前往后接即可）；23. 合并 K 个升序链表；88 也是归并排序 `merge` 步骤的原型。

---

## 模式四：原地反转技巧

**适用信号**：需要**旋转**数组，或需要「把一段整体挪到另一端而保持各自内部顺序」。核心工具是把数组的一段翻转，而翻转本身用对撞双指针实现。

### 189. 轮转数组（中等）

**题目**：把数组 `nums` 中的元素向右轮转 `k` 个位置，要求原地完成。

**思路（三次反转）**：
直接把每个元素搬到 `(i + k) % n` 会互相覆盖，需要环状替换并小心处理环的起点，代码较绕。三次反转更直观：

1. 整体反转：`[1,2,3,4,5,6,7]` → `[7,6,5,4,3,2,1]`；
2. 反转前 `k` 个：→ `[5,6,7,4,3,2,1]`；
3. 反转后 `n-k` 个：→ `[5,6,7,1,2,3,4]`。

为什么对：整体反转把「要移到前面的后缀」和「要后移的前缀」都倒了过来，再分别把两段各自反转回来，就恢复了各自内部的顺序，同时交换了两段的位置。注意 `k` 先对 `n` 取模（轮转 `n` 次等于没转），也避免 `k > n` 时越界。

**代码**（`src/array/rotate_array.py` / `.cpp`）：

```python
def rotate(nums, k):
    n = len(nums)
    k %= n

    def reverse(lo, hi):
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo += 1
            hi -= 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

```cpp
void rotate(std::vector<int> &nums, int k) {
    int n = static_cast<int>(nums.size());
    if (n == 0) return;
    k %= n;
    std::reverse(nums.begin(), nums.end());
    std::reverse(nums.begin(), nums.begin() + k);
    std::reverse(nums.begin() + k, nums.end());
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`k %= n` 必须做（`k` 可能大于 `n`）；边界情况 `k == 0` 或空数组时三次反转仍成立，但 C++ 里要防 `n == 0` 时取模除零；`reverse(0, k-1)` 当 `k == 0` 时区间为空，正确处理。
- **相似题**：151. 翻转字符串里的单词（先整体翻转再局部翻转的同一思想）；541. 反转字符串 II；344. 反转字符串（本篇「对撞双指针」的字符串版）。

---

## 规律总结

1. **对撞双指针 = 有序或对称 + 两端逼近 + 可证明的排除**。`167` 排除的是「矮端那一侧」，`11` 排除的是「较矮的那根线」，`15` 先排序再降维成 `167`，`125` 则利用「回文左右对称」跳过干扰字符。判断该动哪一端，本质是问「当前这一端还有没有可能出现在最优解里」。
2. **快慢指针 = 保留条件 + 原地写入**。`slow` 是写指针，`fast` 是读指针。26「与前一个不同」、27「不是 val」、80「与倒数第二个不同」、283「非零」，全都是「换一个保留条件」的同一道题；只是去重/移除可以覆盖写，而移动零需要交换来保住相对顺序。
3. **方向不是只有从前往后**。当目标数组尾部有空位时（88），从后往前写反而更简单，因为不会覆盖还没读的数据。先问一句「哪边的空间是富余的」。
4. **反转是一把瑞士军刀**。189 用「整体反转 + 两段各自反转」实现轮转，151 用同样的思路翻单词。翻转本身又是对撞双指针，招式之间是复用的。
5. **排序往往是双指针的前置步骤**：它同时带来「有序可收缩」和「相同元素相邻便于去重」两个好处。
6. 遇到「无序数组两数之和」不要硬套对撞双指针，那要用哈希表（下一篇）。
