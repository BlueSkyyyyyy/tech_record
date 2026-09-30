# 二分查找

二分查找是「有序」这件事能带来的最大红利：只要区间里的元素**按大小排好**，
我们就能不断把搜索范围砍掉一半，把 O(n) 的线性扫描压成 O(log n)。它的思想朴素到一句话：
**每一步都排除掉不可能存在答案的那一半**。

二分的代码很短，却是最容易写错的算法之一。错误几乎都来自同一个地方——**区间的定义**。
只要先想清楚「我要维护的区间是闭的还是开的」，再严格按这个定义去更新 `left` / `right`，
边界就不会出错。本篇先把最基础的一段讲透，后面的「二分答案」等进阶用法都会建立在这套
「区间不变式」上。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 标准二分（查等值） | 704. 二分查找 | 简单 |
| lower_bound（查插入位置） | 35. 搜索插入位置 | 简单 |
| 左右边界（lower/upper bound） | 34. 在排序数组中查找元素的第一个和最后一个位置 | 中等 |

---

## 模式一：标准二分（左闭右闭）

**适用信号**：数组**有序**，要找某个值存不存在、或它的确切下标。关键词是「有序」+「找一个点」。

核心动作：维护一个**闭区间** `[left, right]`，区间里的元素都是「还有可能」的候选。
每轮取中点，和目标比较，把不可能的一半连同中点一起排除掉。
区间用闭区间的语言描述，那么：

- 区间为空的条件是 `left > right`，所以循环写成 `while left <= right`；
- 排除中点时，新边界要跳过它，写成 `left = mid + 1` 或 `right = mid - 1`。

### 704. 二分查找（简单）

**题目**：给定一个升序排列的整数数组 `nums` 和一个目标值 `target`，在数组中查找 `target`。如果存在返回它的下标，否则返回 `-1`。例如 `nums = [-1, 0, 3, 5, 9, 12]`，`target = 9` 返回 `4`，`target = 2` 返回 `-1`。

**思路（每次砍一半）**：
闭区间 `[left, right]` 表示「答案还可能落在这里」。取中点 `mid`，比较 `nums[mid]` 与 `target`：

- 相等：命中，返回 `mid`；
- `nums[mid] < target`：数组有序，说明 `mid` 以及它左边的一切都太小，答案是或不是只可能在右半段，于是 `left = mid + 1`；
- `nums[mid] > target`：对称地，`right = mid - 1`。

为什么用 `mid = left + (right - left) // 2` 而不是 `(left + right) // 2`：
两者在中点位置上等价，但 C++ 里当 `left`、`right` 都接近 `int` 上限时，`left + right` 会**溢出**成负数，
随后 `mid` 取到非法下标。写成「先减后加」就绕开了这次加法溢出的可能。

为什么循环条件是 `left <= right`：闭区间 `[left, right]` 在 `left == right` 时并未为空，
里面还剩恰好一个元素，必须再比一次才能确定它是不是答案。只有当 `left` 越过 `right`
（`left = right + 1`）时区间才真正为空，循环退出、返回 `-1`。

为什么更新要带 `±1`：既然 `mid` 已经比较过、确定不是答案，就必须把它移出候选区间。
若不 `±1`，`mid` 会一直留在区间里，可能出现区间不再缩小而死循环。**「闭区间 + 跳过中点」
是一对不可拆开的约定。**

**代码**（完整可运行版见 `src/binary-search/binary_search.py` / `.cpp`）：

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

```cpp
int search(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：`mid` 必须用 `left + (right - left) // 2` 防溢出；循环条件与界更新要配套（闭区间用 `<=`、界带 `±1`）；`nums` 为空时 `right = -1`，循环直接不进入、返回 `-1`，是正确行为；数组必须真的有序，二分对无序数组的结果无意义。
- **相似题**：35. 搜索插入位置（把「等于」并进右压分支就得到插入位置，见下）；34. 查找左右边界（重复元素时要专门定位边界，见下）；278. 第一个错误的版本（判定条件从「比大小」换成「是否为坏版本」，是二分答案的雏形）；374. 猜数字大小（本质是二分查找）。

---

## 模式二：lower_bound（找第一个大于等于的位置）

**适用信号**：要找「第一个满足某个条件的下标」，而这个条件在数组上**单调**——
一旦某个位置满足，它右边全部满足。常见问法有「插入位置」「第一个不小于 x 的数」「第一个坏版本」。

核心动作：和标准二分只差一处——**把「等于」也归入往左压的分支**。
判定只问一句「`nums[mid]` 是否小于 `target`」：小于就往右走，否则往左压。
循环结束时 `left` 恰好停在「第一个 `>= target` 的位置」，这就是 `lower_bound`。

### 35. 搜索插入位置（简单）

**题目**：给定一个升序排列、元素互不相同的整数数组 `nums` 和目标值 `target`。如果 `target` 存在就返回它的下标；否则返回它按顺序插入后应有的下标。要求 O(log n)。例如 `nums = [1, 3, 5, 6]`，`target = 5` 返回 `2`，`target = 2` 返回 `1`，`target = 7` 返回 `4`。

**思路（收敛到第一个 >= target 的位置）**：
插入位置的定义就是「第一个大于等于 `target` 的元素的下标」——把 `target` 插在它前面，
数组仍然有序。所以本题求的就是 `lower_bound`。

和 704 的关键区别：即便 `nums[mid] == target` 也**不能立即返回**，因为我们要的是最靠左的那个满足位置，
左边也许还有等于 `target` 的元素（本题虽不重复，但这个写法是为通用模板做准备）。
把判定统一成「`nums[mid] < target`」：是则整个左半段都太小，`left = mid + 1`；
否则 `mid` 本身可能是答案、但左边也许更靠前，`right = mid - 1` 继续向左压。

循环结束时 `left > right`，`left` 的值有两种情况：落在 `[0, n-1]` 内（插到某个元素之前），
或等于 `n`（`target` 比所有元素都大，插到末尾）。两者都正好是答案，无需特判。

为什么这个「只问小于」的模板值得单独记：它把「找等于」和「找插入点」统一成同一段代码，
而且天然返回「第一个满足 `>= target`」的位置。后面 34 的左右边界，不过是它的两个变体。

**代码**（`src/binary-search/search_insert.py` / `.cpp`）：

```python
def search_insert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left
```

```cpp
int searchInsert(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：`target` 比所有元素都大时返回 `n`，不能把返回值当成「一定命中」去访问 `nums[返回值]`；判定条件里**不要写 `<=`**，否则会把等于也往右推、得到 upper_bound 而非插入位置；空数组要返回 0，循环不进入、`left = 0` 恰好正确。
- **相似题**：704. 二分查找（同一个模板，遇到等于提前返回，见上）；34. 查找左右边界（本函数就是它的前半截，见下）；278. 第一个错误的版本（`lower_bound` 的判定换成 `isBadVersion`）；162. 寻找峰值、153. 旋转数组的最小值（都靠「和一侧邻居比较」构造单调性，再用二分，属于后续进阶）。
- **补充写法**：也可以在 `a[mid] == target` 时直接返回，因为本题元素不重复；但用统一的 `lower_bound` 模板更省心，不会漏掉「插到末尾」这类边界。

---

## 模式三：左右边界（lower_bound + upper_bound）

**适用信号**：数组**有重复元素**，要定位某个值出现的**一整段**，而不是任意一个位置。
普通二分找到的那个下标会落在重复段中间，边界信息丢失，必须用两个二分分别收左右两侧。

核心动作：定义一对「孪生」函数——

- `lower_bound`：第一个 `>= target` 的下标（判定 `nums[mid] < target` 就往右）；
- `upper_bound`：第一个 `> target` 的下标（判定 `nums[mid] <= target` 就往右）。

两者只差一个等号。`target` 若存在，它的出现区间就是 `[lower_bound, upper_bound - 1]`；
若不存在，则 `lower_bound == upper_bound`（都指向「应该插入的位置」），据此就能判空。

### 34. 在排序数组中查找元素的第一个和最后一个位置（中等）

**题目**：给定一个非递减排列的整数数组 `nums`（可能含重复元素）和目标值 `target`，找出 `target` 出现的起始下标和结束下标；不存在则返回 `[-1, -1]`。要求 O(log n)。例如 `nums = [5, 7, 7, 8, 8, 10]`，`target = 8` 返回 `[3, 4]`，`target = 6` 返回 `[-1, -1]`。

**思路（两次二分分别收边界）**：
先求 `first = lower_bound(nums, target)`。它是第一个 `>= target` 的位置。
如果 `first` 已经越界（等于 `n`），或它指向的元素不等于 `target`，说明数组里根本没有 `target`，
直接返回 `[-1, -1]`；否则 `first` 就是左边界。
再求 `upper_bound(nums, target)`，它指向第一个 `> target` 的位置，减一就是最后一个等于 `target` 的下标，
即右边界。

为什么不能「先用普通二分找到 `target`，再向两边线性扩展」：重复元素可能很多，
那一步最坏是 O(n)，会把整体复杂度从 O(log n) 拖回 O(n)，违背题目要求。
两次二分各自 O(log n)，合计仍是 O(log n)。

为什么 `lower_bound` 和 `upper_bound` 只差一个等号：两者都是「把满足判定的一段往右排除」。
`lower_bound` 要排除「小于 target」的，所以判定用 `<`；`upper_bound` 要多排除「等于 target」的，
所以判定用 `<=`。把这个等号记牢，两个函数就能互相推导。

**代码**（`src/binary-search/find_first_and_last.py` / `.cpp`）：

```python
def lower_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def upper_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid - 1
    return left


def search_range(nums, target):
    first = lower_bound(nums, target)
    if first == len(nums) or nums[first] != target:
        return [-1, -1]
    last = upper_bound(nums, target) - 1
    return [first, last]
```

```cpp
int lowerBound(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}

int upperBound(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] <= target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left;
}

std::vector<int> searchRange(const std::vector<int> &nums, int target) {
    int first = lowerBound(nums, target);
    if (first == static_cast<int>(nums.size()) || nums[first] != target)
        return {-1, -1};
    int last = upperBound(nums, target) - 1;
    return {first, last};
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：判存在用的是 `first == n || nums[first] != target`，三个条件缺一不可——漏掉越界检查会在空数组或目标过大时越界访问 `nums[first]`；`upper_bound` 返回的是「第一个大于」，右边界要减一；别用 `upper_bound(target + 1)` 代替 `upper_bound(target)`，当 `target` 是 `INT_MAX` 时 `target + 1` 会溢出；`target` 不存在时返回 `[-1, -1]`，不要返回 `[lower, lower-1]` 这种非法区间。
- **相似题**：704. 二分查找（基础模板，见上）；35. 搜索插入位置（本函数里的 `lower_bound` 单独使用，见上）；278. 第一个错误的版本；在 300. 最长递增子序列 的 `O(n log n)` 解法里，`lower_bound` 用来更新「tails 数组」，是它的经典应用。

---

## 规律总结

1. **二分的难点不是「折半」，而是「区间定义」**。动手前先明确：我维护的是闭区间 `[left, right]` 还是左闭右开 `[left, right)`？本篇统一用**闭区间**：循环条件 `while left <= right`，排除中点时 `left = mid + 1` / `right = mid - 1`，两者必须配套。改用左闭右开时，条件变成 `while left < right`、`right = mid`（不减一），同样要整套一起改，切忌混用。
2. **`mid` 一律写 `left + (right - left) // 2`**，避免 `left + right` 在 C++ 里溢出。这是二分的标准防身写法。
3. **「找等值」和「找位置」只差一个分支**。标准二分遇到等于提前返回；把「等于」并进「往左压」的分支（只判定 `<`），循环收敛到的 `left` 就是第一个 `>= target` 的位置，也就是 `lower_bound`／插入位置。
4. **`lower_bound` 与 `upper_bound` 只差一个等号**。前者判定 `nums[mid] < target`，收敛到第一个 `>=`；后者判定 `nums[mid] <= target`，收敛到第一个 `>`。有重复元素时用这两个函数夹出 `[lower, upper-1]` 这段区间，别用「命中后线性扩展」，那会退化成 O(n)。
5. **先判存在、再取边界**。求左右边界前，先检查 `lower_bound` 是否越界或所指元素不等于目标；否则空数组、目标过大等情况会越界访问。
6. **二分的判定条件必须具有单调性**。「第一个满足某条件的下标」这类问题之所以能二分，是因为条件在某点之前为假、之后全为真。数组有序只是这种单调性的最常见特例；像 278 的「坏版本」、162 的「峰值」、以及后续的「二分答案」，都是在构造并使用这种单调性。
7. **以 `src` 为准，docs 只做粘贴**。题解里的代码必须和 `code/leetcode/src/binary-search/` 下的实现**逐字一致**，避免文档与可运行代码脱节。
