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
| 旋转有序数组中查找 | 33. 搜索旋转排序数组 | 中等 |
| 旋转数组的最小值 | 153 / 154. 寻找旋转排序数组中的最小值 I / II | 中等 / 困难 |
| 寻找峰值（局部单调性） | 162. 寻找峰值 | 中等 |
| 二分答案 | 69. x 的平方根 | 简单 |
| 二分答案 | 875. 爱吃香蕉的珂珂 | 中等 |
| 二分答案 | 1011. 在 D 天内送达包裹的能力 | 中等 |

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

## 模式四：旋转有序数组中的查找

**适用信号**：数组本来是升序的，但在某个位置被「旋转」（前后两段互换），要在其中找一个值。
关键词是「有序但被切开」——整段不再有序，但**切一刀后总有一半是有序的**。

核心动作：每轮取中点，先判断哪一半是有序的，再看目标是否落在这一半的范围内；落在里面就去这半找，
否则去另一半。判断「有序」只需比较 `nums[left]` 与 `nums[mid]`。

### 33. 搜索旋转排序数组（中等）

**题目**：整数数组 `nums` 原本升序且元素互不相同，但在某个未知下标被旋转（如 `[0,1,2,4,5,6,7]` 变成 `[4,5,6,7,0,1,2]`）。给定 `target`，返回它的下标，不存在返回 `-1`。要求 O(log n)。例如 `nums = [4,5,6,7,0,1,2]`，`target = 0` 返回 `4`，`target = 3` 返回 `-1`。

**思路（先找有序的一半）**：
旋转相当于把一个升序数组从中间某处剪断再拼接，所以整体看是「先升后降再升」的两段。
关键性质是：**对任意 `mid`，`[left, mid]` 与 `[mid, right]` 中至少有一段是完全升序的。**

于是每轮取 `mid` 后分两种情况：

- `nums[left] <= nums[mid]`：左半段 `[left, mid]` 有序。看 `target` 是否满足 `nums[left] <= target < nums[mid]`：
  是，则答案只可能在这段里，`right = mid - 1`；否则答案在右半段，`left = mid + 1`。
- 否则：右半段 `[mid, right]` 有序。看 `target` 是否满足 `nums[mid] < target <= nums[right]`：
  是则 `left = mid + 1`，否则 `right = mid - 1`。

为什么「判断出有序的那一半后，就能整段排除另一半」：在有序段里，范围比较是可靠的——
目标若不落在它的值域内，就绝不会出现在这段；而另一半虽然无序，但既然这段被排除，答案只能去那里。
每一轮都精确排除一半，因此复杂度仍是 O(log n)。

为什么用 `nums[left] <= nums[mid]` 而不是 `<`：当区间收窄到 `left == mid`（只有一个元素）时，
仍然要把它归入「左半有序」分支，否则会漏判。

为什么判断区间时 `mid` 是开边界：`nums[mid]` 在循环开头已经比较过，若不等于 `target` 就必然不是答案，
所以左半写成 `target < nums[mid]`（不含 `mid`），右半写成 `nums[mid] < target`（不含 `mid`）。

**代码**（`src/binary-search/search_in_rotated_sorted_array.py` / `.cpp`）：

```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

```cpp
int searchRotated(const std::vector<int> &nums, int target) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }
    return -1;
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：两段范围判断里的 `mid` 必须开边界（`target < nums[mid]` / `nums[mid] < target`），写成闭边界会在 `target` 恰好落在边界附近时把正确答案排除；`nums[left] <= nums[mid]` 的等号不能丢，否则 `left == mid` 时会走错分支；本题元素互不相同，若允许重复（81 题）需要额外处理 `nums[left] == nums[mid]` 的情况。
- **相似题**：153 / 154. 寻找旋转排序数组中的最小值（同族但换了个问法，见下）；704. 二分查找（未旋转时的退化情形，见上）；81. 搜索旋转排序数组 II（允许重复元素的版本，遇到左右相等时只能移动一端）。

---

## 模式五：旋转数组的最小值

**适用信号**：同样是被旋转的升序数组，但要找的是**最小值**（分界点），而不是某个具体值。
问法有「旋转数组最小值」「第一个小于末尾的数」等。

核心动作：固定**拿 `nums[mid]` 和右端点 `nums[right]` 比较**：

- `nums[mid] > nums[right]`：`mid` 在较大的左半段，最小值一定在 `mid` 右侧，`left = mid + 1`；
- `nums[mid] < nums[right]`：从 `mid` 到 `right` 递增，最小值在 `mid` 或其左，`right = mid`。

循环用 `while left < right`，退出时 `left == right` 即最小值下标。有重复元素时，`nums[mid] == nums[right]`
无法判断方向，只能保守地 `right -= 1`。

### 153. 寻找旋转排序数组中的最小值（中等）

**题目**：一个原本升序、元素互不相同的数组在未知下标被旋转（如 `[3,4,5,1,2]`）。返回最小元素。要求 O(log n)。例如 `[3,4,5,1,2]` 返回 `1`，`[4,5,6,7,0,1,2]` 返回 `0`。

**思路（比较中点与右端）**：
旋转后的数组是一个「先升到最大、再掉到最小、再升回去」的形状，最小值就是那个掉下去的位置。
用 `nums[mid]` 与 `nums[right]` 比较来锁定它在哪一侧：

- 若 `nums[mid] > nums[right]`：说明 `mid` 落在较大的上升段，最小值不可能在 `mid` 及它左边，`left = mid + 1`；
- 若 `nums[mid] < nums[right]`：说明区间 `[mid, right]` 是递增的，最小值在 `mid` 或 `mid` 左边，`right = mid`。

为什么 `right = mid` 而不是 `mid - 1`：`nums[mid]` 本身**有可能就是最小值**（比如 `[1,2]` 的 `mid = 0`，
`nums[0] < nums[1]`，最小值就是下标 0），所以要把它留在候选区间里。

为什么和右端比而不是左端：旋转的「断点」信息在右端更稳定——`nums[mid] > nums[right]` 一定意味着
最小值在右侧，判断简单可靠。固定「和右端比」是一套记忆负担最小的模板。

为什么循环用 `while left < right`：这里维护的是一个会收缩到单点的区间，`left == right` 时答案已确定，
不需要再比一次；这也和「`right = mid` 不跳过」的更新配套，两者是同一套闭区间收缩约定。

**代码**（`src/binary-search/find_minimum_in_rotated_sorted_array.py` / `.cpp`）：

```python
def find_min(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]
```

```cpp
int findMin(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right])
            left = mid + 1;
        else
            right = mid;
    }
    return nums[left];
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：`right = mid` 不能写成 `right = mid - 1`，否则可能把最小值本身排除；循环必须用 `while left < right`，若用 `<=` 且 `right = mid` 会死循环；数组未旋转（仍升序）时也能正确返回 `nums[0]`，无需特判；返回的是 `nums[left]` 而不是 `left`。
- **相似题**：154. 寻找旋转排序数组中的最小值 II（允许重复元素，见下）；33. 搜索旋转排序数组（同族但找具体值，见上）；162. 寻找峰值（都靠「和邻居比较」构造方向，见下）。

### 154. 寻找旋转排序数组中的最小值 II（困难）

**题目**：和 153 相同，但数组中**允许包含重复元素**，返回最小元素。

**思路（相等时保守右移）**：
沿用 153「和右端比」的框架，但重复元素会带来一个新情况：当 `nums[mid] == nums[right]` 时，
无法判断最小值在 `mid` 的哪一侧。例如 `[1,1,1,0,1]` 里 `mid` 和 `right` 可能都是 1，最小值 0 在右侧；
而 `[1,0,1,1,1]` 同样两端都是 1，最小值却在左侧。方向不可判定，二分的信息在这里被「抹平」了。

这时唯一不会丢解的做法是 `right -= 1`：既然 `nums[right]` 等于 `nums[mid]`，而 `mid < right`，
那么即使最小值恰好等于这个值，`mid` 位置也已经保留了它，把最右端这个重复元素丢掉是安全的。

这样做的代价是：当数组里有大量重复（极端情况全是相同元素）时，每轮只缩一格，退化成 O(n)。
这是含重复元素下无法避免的最坏情况；元素互不相同时，它自动退化为 153 的 O(log n)。

**代码**（`src/binary-search/find_minimum_in_rotated_sorted_array_ii.py` / `.cpp`）：

```python
def find_min_ii(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]
```

```cpp
int findMinII(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[right])
            left = mid + 1;
        else if (nums[mid] < nums[right])
            right = mid;
        else
            right -= 1;
    }
    return nums[left];
}
```

- **复杂度**：平均/最好 O(log n)，最坏（大量重复元素）O(n)，空间 O(1)。
- **易错点**：相等分支只能是 `right -= 1`，不能把 `right = mid` 或 `left = mid + 1` 硬套进去（方向未知，会漏解）；复杂度退化是重复元素本身的固有代价，不必也不应强行写成严格 O(log n)；判断仍必须以 `right` 端为基准，和 153 保持一致。
- **相似题**：153. 寻找旋转排序数组中的最小值（无重复版本，是本题的特例，见上）；81. 搜索旋转排序数组 II（同样因重复元素在「无法判断」时退让一步）。

---

## 模式六：寻找峰值（局部单调性）

**适用信号**：数组无序，但要找**局部极值**（比左右邻居都大的点），并且允许返回任意一个。
关键词是「任意峰值」+「相邻比较」。

核心动作：比较 `nums[mid]` 与 `nums[mid+1]`，**往更高的一侧走**：

- `nums[mid] < nums[mid+1]`：上坡，右侧必有峰值，`left = mid + 1`；
- `nums[mid] > nums[mid+1]`：下坡，`mid` 或其左侧必有峰值，`right = mid`。

### 162. 寻找峰值（中等）

**题目**：峰值元素是严格大于左右相邻元素的元素。给定整数数组 `nums`（可能含多个峰值），返回任意一个峰值的下标。约定 `nums[-1] = nums[n] = -∞`。要求 O(log n)。例如 `[1,2,3,1]` 的峰值下标是 2，`[1,2,1,3,5,6,4]` 的峰值是下标 1 或 5。

**思路（向更高处走）**：
峰值问题表面上和「有序」无关，二分为何还能用？靠的是一种**局部单调性**：任意位置只要向右看一步，
就能知道「哪边一定藏着峰」。比较 `nums[mid]` 和 `nums[mid+1]`：

- 若 `nums[mid] < nums[mid+1]`：正在上坡，那就一直往右走——要么右端某处由升转降，那个转折点就是峰；
  要么一路升到数组末尾，而 `nums[n] = -∞`，末尾元素自然成为峰。所以右侧必有峰，`left = mid + 1`。
- 若 `nums[mid] > nums[mid+1]`：正在下坡，`mid` 本身可能就是峰，或者左边某处由升转降成为峰，
  所以 `right = mid`，把 `mid` 留在候选里。

这套「沿着上坡走必到峰值」的论证，就是二分所需的单调性来源。`while left < right` 结束时，
`left == right` 收敛到一个峰值下标即可返回，本题不要求是哪一个。

为什么不用管左边：题目只要求返回任意一个峰值，向任意一侧的上坡方向走都必然撞到一个峰，
两边都看反而复杂。

**代码**（`src/binary-search/find_peak_element.py` / `.cpp`）：

```python
def find_peak_element(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left
```

```cpp
int findPeakElement(const std::vector<int> &nums) {
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] > nums[mid + 1])
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}
```

- **复杂度**：时间 O(log n)，空间 O(1)。
- **易错点**：访问 `nums[mid+1]` 是安全的——循环条件 `left < right` 保证 `mid < right <= n-1`；`right = mid` 不能写成 `mid - 1`，因为 `mid` 本身可能是峰；返回值是下标，不是元素值；题目允许返回任意峰值，测试时不要死盯某个特定下标（`[1,2,1,3,5,6,4]` 返回 1 或 5 都对）。
- **相似题**：153 / 154. 旋转数组最小值（都靠「和邻居比较」定方向，见上）；852. 山脉数组的峰顶索引（保证只有一个峰的简化版）；1095. 山脉数组中查找目标值（先用本题思路找峰，再在两侧二分）。

---

## 模式七：二分答案（猜答案 + 单调验证）

**适用信号**：题目问「最小/最大的某个量」，这个量本身**不好直接算**，但你可以快速验证
「某个候选值行不行」；并且验证结果随候选值**单调变化**（越大越容易满足，或越小越容易满足）。
这时就二分答案的取值区间。常见特征词：「最小速度」「最少天数」「最小载重」「最大……的最小值」。

核心动作分两步，缺一不可：

1. **写验证函数** `feasible(x)`：给定候选答案，判断它是否可行（通常一次 O(n) 扫描或贪心）。
2. **在答案值域上二分**：确定 `[lo, hi]`，用 `while left < right` 找**第一个** `feasible` 为真的点；
   `feasible(mid)` 为真则 `right = mid`（保留 mid），否则 `left = mid + 1`。

这套模板与「在数组里二分」的唯一区别是：二分对象从**下标**换成了**答案本身**。写好验证函数、
定对上下界，剩下的就是套用闭区间收缩。

### 69. x 的平方根（简单）

**题目**：给定非负整数 `x`，返回它的算术平方根的整数部分（向下取整），不能用内置指数/开方函数。例如 `x = 4` 返回 `2`，`x = 8` 返回 `2`。

**思路（在 [0, x] 上二分最大的 k 使 k² ≤ x）**：
要找的是「最大的满足 `k*k <= x` 的整数 k」。验证函数就是 `k*k <= x`，它随 k 增大由真变假，
单调，因此可以二分。

维护闭区间 `[left, right]`，其中 `left`/`right` 是**答案 k 的候选值域**（不是数组下标）：

- `mid*mid <= x`：`mid` 可行，也许还有更大的可行值，`left = mid + 1`；
- 否则：`mid` 太大，`right = mid - 1`。

循环结束时 `right` 恰好是最后一个可行值，返回它（等价地也可返回 `left - 1`）。

为什么右边界取 `x` 而不是 `x // 2`：`x` 很小时（`x = 0` 或 `1`）`x // 2` 会小于真实答案，
统一取 `x` 更稳妥，多出的对数级次数可以忽略。为什么 C++ 里 `mid*mid` 要提成 `long long`：
`mid` 最大接近 2³¹，平方会溢出 32 位 `int`，必须用 `1LL * mid * mid` 先升到 64 位。

**代码**（`src/binary-search/sqrtx.py` / `.cpp`）：

```python
def my_sqrt(x):
    left, right = 0, x
    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid <= x:
            left = mid + 1
        else:
            right = mid - 1
    return left - 1
```

```cpp
int mySqrt(int x) {
    int left = 0, right = x;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (1LL * mid * mid <= x)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return left - 1;
}
```

- **复杂度**：时间 O(log x)，空间 O(1)。
- **易错点**：C++ 里 `mid*mid` 必须用 `long long`，否则 `x` 接近 `INT_MAX` 时溢出；本题用的是 `while left <= right` 加 `±1` 的闭区间模板（找「最后一个可行值」），和下面 875/1011 常用的 `while left < right` 模板不同，两者别混用；`x = 0` 时返回 `left - 1 = 0`，正确。
- **相似题**：367. 有效的完全平方数（判断是否存在 k 使 k² = x）；875 / 1011（同为二分答案，见下）。

### 875. 爱吃香蕉的珂珂（中等）

**题目**：有 `n` 堆香蕉，第 `i` 堆 `piles[i]` 根。珂珂每小时选一堆吃 `k` 根，不足 `k` 根时吃完这堆后本小时不再吃别的。警卫将在 `h` 小时后回来（`h >= piles.length`），求能吃完所有香蕉的最小速度 `k`。

**思路（猜速度，验证总耗时）**：
与其纠结每小时怎么分配，不如直接**猜速度 `k`**，再验证它够不够快。关键观察：速度越大、总耗时越少，
「能否在 h 小时内吃完」这个判定随 k 增大从假变真，存在临界点，正是答案——这就是二分的单调性。

验证函数 `can_finish(k)`：第 `i` 堆需要 `ceil(piles[i] / k)` 小时，整数上取整写成
`(piles[i] + k - 1) // k`；把各堆耗时相加，不超过 `h` 即成功。

答案区间取 `[1, max(piles)]`：下界是 1（不能不吃），上界是最大堆容量——一小时最多吃一堆，
速度再大也不会更快。在这个区间上二分第一个可行的 k（`while left < right` 收缩到单点）。

**代码**（`src/binary-search/koko_eating_bananas.py` / `.cpp`）：

```python
def min_eating_speed(piles, h):
    def can_finish(k):
        hours = 0
        for pile in piles:
            hours += (pile + k - 1) // k
        return hours <= h

    left, right = 1, max(piles)
    while left < right:
        mid = left + (right - left) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

```cpp
bool canFinish(const std::vector<int> &piles, int h, int k) {
    long long hours = 0;
    for (int pile : piles) hours += (pile + k - 1) / k;
    return hours <= h;
}

int minEatingSpeed(const std::vector<int> &piles, int h) {
    int left = 1, right = 0;
    for (int pile : piles) right = std::max(right, pile);
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canFinish(piles, h, mid))
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}
```

- **复杂度**：时间 O(n log(max(piles)))，空间 O(1)。
- **易错点**：上取整必须写成 `(pile + k - 1) // k`，直接整除会少算时间导致答案偏小；C++ 里累计耗时用 `long long` 防溢出；答案下界是 1 而非 0（`k = 0` 会除零）；`feasible` 为真时用 `right = mid` 保留 mid，找的是**最小**可行速度。
- **相似题**：1011. 在 D 天内送达包裹的能力（同一模板，见下）；69. x 的平方根（更简单的二分答案，见上）；410. 分割数组的最大值（把「载重」换成「子数组和上界」）；1482. 制作 m 束花所需的最少天数（同为「最小可行阈值」）。

### 1011. 在 D 天内送达包裹的能力（中等）

**题目**：包裹必须**按给定顺序**装载，船每天装一个连续区间的包裹，总重量不超过载重 `capacity`。求能在 `days` 天内运完所有包裹的最小载重。

**思路（猜载重，贪心验证天数）**：
和 875 是同一套模板，只是验证函数换成了「按顺序贪心分组」。不去枚举怎么切分，而是**猜载重 `capacity`**：
载重越大、需要天数越少，「能在 days 天内运完」随载重增大由假变真，二分临界点即可。

验证函数 `can_ship(cap)`：按顺序扫描包裹，能塞进当前船就塞，塞不下就开新的一天；
最后看使用天数是否不超过 `days`。这个贪心是对的——顺序固定时，每天尽量多装不会让后续更差。

答案区间取 `[max(weights), sum(weights)]`：下界是最大单个包裹（载重再小就装不下它），
上界是所有包裹总重（一天全运走必然可行）。在区间上二分第一个可行的载重。

为什么和 875 归为一类：两者都是「最小化阈值，使可行性判定成立」，区别只在验证函数。
遇到这类题，先想清楚「要二分的量是什么」和「怎么 O(n) 验证」，代码结构就自动浮现了。

**代码**（`src/binary-search/capacity_to_ship_packages_within_d_days.py` / `.cpp`）：

```python
def ship_within_days(weights, days):
    def can_ship(cap):
        used_days = 1
        cur = 0
        for w in weights:
            if cur + w > cap:
                used_days += 1
                cur = 0
            cur += w
        return used_days <= days

    left, right = max(weights), sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    return left
```

```cpp
bool canShip(const std::vector<int> &weights, int days, int cap) {
    int used = 1, cur = 0;
    for (int w : weights) {
        if (cur + w > cap) {
            used += 1;
            cur = 0;
        }
        cur += w;
    }
    return used <= days;
}

int shipWithinDays(const std::vector<int> &weights, int days) {
    int left = 0, right = 0;
    for (int w : weights) {
        left = std::max(left, w);
        right += w;
    }
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canShip(weights, days, mid))
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}
```

- **复杂度**：时间 O(n log(sum(weights)))，空间 O(1)。
- **易错点**：答案下界必须是 `max(weights)`，取 1 或 0 会让 `canShip` 对大包裹判定失败甚至除零式错误；`used` 初始化为 1（至少用一天），遍历时先判断「当前船 + 本包裹是否超载」再决定是否开新天；C++ 里 `right = sum(weights)` 用 `int` 足够（LeetCode 范围内），但大范围数据宜用 `long long`；`feasible` 真时 `right = mid` 才能取到最小载重。
- **相似题**：875. 爱吃香蕉的珂珂（同一模板的孪生题，见上）；410. 分割数组的最大值（把「天」换成「段数 m」）；774. 最小化去加油站的最大距离（浮点二分答案）；69. x 的平方根（最简二分答案，见上）。

---

## 规律总结

1. **二分的难点不是「折半」，而是「区间定义」**。动手前先明确：我维护的是闭区间 `[left, right]` 还是左闭右开 `[left, right)`？本篇统一用**闭区间**：循环条件 `while left <= right`，排除中点时 `left = mid + 1` / `right = mid - 1`，两者必须配套。改用左闭右开时，条件变成 `while left < right`、`right = mid`（不减一），同样要整套一起改，切忌混用。
2. **`mid` 一律写 `left + (right - left) // 2`**，避免 `left + right` 在 C++ 里溢出。这是二分的标准防身写法。
3. **「找等值」和「找位置」只差一个分支**。标准二分遇到等于提前返回；把「等于」并进「往左压」的分支（只判定 `<`），循环收敛到的 `left` 就是第一个 `>= target` 的位置，也就是 `lower_bound`／插入位置。
4. **`lower_bound` 与 `upper_bound` 只差一个等号**。前者判定 `nums[mid] < target`，收敛到第一个 `>=`；后者判定 `nums[mid] <= target`，收敛到第一个 `>`。有重复元素时用这两个函数夹出 `[lower, upper-1]` 这段区间，别用「命中后线性扩展」，那会退化成 O(n)。
5. **先判存在、再取边界**。求左右边界前，先检查 `lower_bound` 是否越界或所指元素不等于目标；否则空数组、目标过大等情况会越界访问。
6. **二分的真正前提不是「数组有序」，而是「判定具有单调性」**。「第一个满足某条件的下标」这类问题之所以能二分，是因为条件在某点之前为假、之后全为真。数组有序只是这种单调性的最常见特例。局部单调性（162 的峰值：往高处走必有峰）和答案值域上的单调性（二分答案），都是同一件事的不同外衣。判断一道题能不能二分，先问自己：**能不能找到一个「前假后真」的判定条件？**
7. **旋转数组两步走：先找有序的一半，再决定去哪半**。对 33 这类「有序但被切开」的题，每轮比较 `nums[left]` 与 `nums[mid]` 判断出哪半有序，再用范围比较把目标定位到其中一半。找最小值时则换一套固定套路——**始终和中点与右端比较**，`nums[mid] > nums[right]` 就往右、否则往左收，把方向判断简化到极致。
8. **区间收缩会「回到单点」时用 `while left < right`**。34、153、154、162 以及 875/1011 都采用这个形式：`left == right` 时区间只剩一个候选，正是答案，无需再比。与之配套的是「命中侧不跳过」——`right = mid`（保留 mid），否则可能把答案本身排除。它和前面「`while left <= right` + `±1`」的模板等价，只是把「找最后一个可行值」改写成了「找第一个可行值」，两套都背下来，按题目问法选用。
9. **二分答案 = 猜答案 + 单调验证**。凡是问「最小/最大的某个量」而直接计算很麻烦的题（69/875/1011），都先写一个 O(n) 的验证函数 `feasible(x)`，再在答案的**值域**上二分。要点有三：验证函数要正确且够快；二分对象是答案本身而非下标；上下界要卡准（下界要保证验证有意义，上界要保证一定可行），否则会死在边界上。
10. **写 C++ 版时留意溢出与宏坑**。`mid` 用 `left + (right - left) / 2`；平方、乘积、累加这类运算按需提成 `long long`（如 69 的 `mid*mid`、875 的累计耗时）。另外，用 `assert` 自测时不要把 `{1, 2}` 这类初值列表直接塞进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先存进变量再比较即可。
11. **以 `src` 为准，docs 只做粘贴**。题解里的代码必须和 `code/leetcode/src/binary-search/` 下的实现**逐字一致**，避免文档与可运行代码脱节。
