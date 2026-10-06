# 随机化与采样：洗牌、加权抽样、水塘抽样与拒绝采样

有一类题不考「最优」，而考**等概率**：如何把随机数「加工」成题目要求的分布。乍看每道题
都不一样，其实底层只有几块积木：

1. **洗牌（Fisher-Yates）**：在数组上原地做 n 次「与剩余区间的随机位置交换」，让 n! 种
   排列等概率出现。
2. **加权抽样**：把权重摊成前缀和数轴，随机取一个点、二分定位落在哪一段。
3. **拒绝采样**：先在一个容易均匀取样的更大范围里取，落在合法区就接受，否则重抽；
   简单但严格均匀。
4. **水塘抽样**：边扫描边以 `1/i` 的概率替换答案，一次遍历、O(1) 空间地等概率取一个。
5. **随机容器**：动态数组负责「等概率取」，哈希表负责「O(1) 定位/删除」，两者拼起来。

本篇收 10 道经典题，按五种套路分组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：费雪-耶茨洗牌 | 384. 打乱数组 / 519. 随机翻转矩阵 | 中等 / 中等 |
| 模式二：按权重抽样（前缀和 + 二分） | 528. 按权重随机选择 / 497. 非重叠矩形中的随机点 | 中等 / 中等 |
| 模式三：拒绝采样与黑名单映射 | 710. 黑名单中的随机数 / 470. 用 Rand7() 实现 Rand10() / 478. 在圆内随机生成点 | 困难 / 中等 / 中等 |
| 模式四：水塘抽样（一次遍历等概率） | 382. 链表随机节点 / 398. 随机数索引 | 中等 / 中等 |
| 模式五：随机容器（数组 + 下标表） | 381. O(1) 插入删除获取随机元素 - 允许重复 | 困难 |

> 「等概率」是可以用数学验证的，而不是靠感觉。写完后不妨问一句：**目标集合里的每个元素，
> 被选中的概率是不是同一个数？** 每一步的随机选择如果都各自均匀、又互不干扰，结论才成立。
> 这与第 16 篇「位运算」里「每个元素各算一次贡献」的计数思路是相通的：随机不等于乱来，
> 分布要能算清楚。

---

## 模式一：费雪-耶茨洗牌

**适用信号**：要求「打乱数组」「等概率随机排列」「随机地逐次取出一个未用过的元素」。

**核心动作**：从后往前，第 i 位与 `[0, i]` 里随机一个位置交换。关键是「剩余区间」在缩小，
且**可以与自己交换**。这样每一步都在「还没排定的位置」里等概率挑一个，保证每个排列
概率都是 1/n!。

### 384. 打乱数组（中等）

**题目**：实现 `Solution`：`reset()` 恢复初始数组，`shuffle()` 等概率随机打乱数组并返回。

**思路**：

```python
class Solution:
    def __init__(self, nums):
        self.original = list(nums)
        self.nums = list(nums)

    def reset(self):
        self.nums = list(self.original)
        return self.nums

    def shuffle(self):
        for i in range(len(self.nums) - 1, 0, -1):
            j = random.randint(0, i)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums
```

```cpp
class Solution {
public:
    Solution(std::vector<int> nums) : original_(nums), nums_(nums) {}

    std::vector<int> reset() {
        nums_ = original_;
        return nums_;
    }

    std::vector<int> shuffle() {
        for (int i = static_cast<int>(nums_.size()) - 1; i > 0; --i) {
            int j = std::rand() % (i + 1);
            std::swap(nums_[i], nums_[j]);
        }
        return nums_;
    }

private:
    std::vector<int> original_;
    std::vector<int> nums_;
};
```

**为什么从后往前**：把「洗牌」想成「依次决定第 n-1、n-2、… 位放谁」。第 n-1 位从全部 n 个
元素里等概率挑（与 `[0, n-1]` 中随机一位交换即可），第 n-2 位从剩下的 n-1 个里等概率挑
（与 `[0, n-2]` 中随机一位交换），……每步概率相乘仍是 1/n!，所以每个排列等概率。若改成
「每次在整个数组里随机选两个位置交换」，同一步可能反复交换同一对元素，分布会偏。

`reset` 单独存一份 `original`，因为 `shuffle` 会就地改动 `nums`；每次 `reset` 重新拷贝，
保证再洗牌仍从初始状态出发。

- **复杂度**：`reset` 时间 O(n)、空间 O(n)；`shuffle` 时间 O(n)、空间 O(1)（不计返回数组）。
- **易错点**：随机范围必须是 `[0, i]`（含 i），写成 `[0, i-1]` 或从前往后都会破坏均匀性；
  Python 里注意 `self.nums` 被就地修改，若外部还需要原序，用 `original` 拷贝。
- **相似题**：519. 随机翻转矩阵（同样用「与末尾交换」的思想维护一个不断缩小的可用集合）。

### 519. 随机翻转矩阵（中等）

**题目**：m×n 的全 0 矩阵，`flip()` 等概率随机选一个 0 格置 1 并返回 `[row, col]`，
`reset()` 全部复位。

**思路**：

```python
class Solution:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.total = m * n
        self.mapping = {}

    def flip(self):
        self.total -= 1
        idx = random.randint(0, self.total)
        chosen = self.mapping.get(idx, idx)
        self.mapping[idx] = self.mapping.get(self.total, self.total)
        return [chosen // self.n, chosen % self.n]

    def reset(self):
        self.total = self.m * self.n
        self.mapping = {}
```

```cpp
class Solution {
public:
    Solution(int m, int n) : m_(m), n_(n), total_(m * n) {}

    std::vector<int> flip() {
        --total_;
        int idx = std::rand() % (total_ + 1);
        auto it = mapping_.find(idx);
        int chosen = it == mapping_.end() ? idx : it->second;
        auto it2 = mapping_.find(total_);
        mapping_[idx] = it2 == mapping_.end() ? total_ : it2->second;
        return {chosen / n_, chosen % n_};
    }

    void reset() {
        total_ = m_ * n_;
        mapping_.clear();
    }

private:
    int m_;
    int n_;
    int total_;
    std::unordered_map<int, int> mapping_;
};
```

**为什么这样映射**：如果矩阵很小、可以开一个 `arr[total]` 记录每个格子的状态，问题就是
384 的洗牌：维护「还剩 `total` 个 0」，每次在 `[0, total-1]` 里随机取一个，把它标记为已用，
再和「第 total-1 个」交换。但矩阵可能很大，不能真开数组，于是用哈希表 `mapping` **只记录
被换动过的编号**：编号 `idx` 如果没被记过就代表它自己；被记过就代表「原本在别处的格子搬到
了这里」。`mapping[idx] = mapping.get(total, total)` 就是把当前末尾编号填到 `idx` 的坑里，
于是可用集合始终是编号 `[0, total)`，取法等价于洗牌。`reset` 只需清空哈希表。
注意「最后一个可用编号」在取完 `idx` 后正好是 `total`（已自减），用它填坑。

- **复杂度**：初始化 O(1)；`flip` / `reset` 平均 O(1)；空间 O(被翻转过的不同格子数)。
- **易错点**：先自减 `total` 再取随机；填坑用的是自减后的 `total`；`reset` 不要把
  `mapping` 漏清空。行列为 `chosen // n` 与 `chosen % n`，别把 m、n 写反。
- **相似题**：384. 打乱数组（同一块「与末尾交换」的洗牌积木）；381. 随机容器（数组 +
  哈希表维护「可用集合」）。

---

## 模式二：按权重抽样

**适用信号**：不是每个元素等概率，而是**按权重/面积比例**；且权重固定、会被多次查询。

**核心动作**：把权重摊成前缀和数轴，随机取 `target ∈ [1, 总权重]`，二分找出它落在哪一段。

### 528. 按权重随机选择（中等）

**题目**：`w[i]` 是下标 i 的权重，实现 `pickIndex()`，使返回下标 i 的概率为
`w[i] / sum(w)`。

**思路**：

```python
class Solution:
    def __init__(self, w):
        self.prefix = []
        total = 0
        for x in w:
            total += x
            self.prefix.append(total)

    def pickIndex(self):
        target = random.randint(1, self.prefix[-1])
        return bisect.bisect_left(self.prefix, target)
```

```cpp
class Solution {
public:
    Solution(std::vector<int> w) {
        int total = 0;
        for (int x : w) {
            total += x;
            prefix_.push_back(total);
        }
    }

    int pickIndex() {
        int target = std::rand() % prefix_.back() + 1;
        return static_cast<int>(
            std::lower_bound(prefix_.begin(), prefix_.end(), target) -
            prefix_.begin());
    }

private:
    std::vector<int> prefix_;
};
```

**为什么这样就能加权**：想象一条长 `sum(w)` 的数轴，下标 i 占据长度为 `w[i]` 的一段
（按前缀和切开）。在数轴上等概率取整数点，点落进哪一段，就返回哪一段的下标；段越长，
落进去的概率越大，恰好是 `w[i] / sum(w)`。前缀和 `prefix[i]` 是第 i 段的右端点，于是
「第一个 `prefix[i] >= target` 的 i」就是答案，用 lower_bound 二分比顺序扫描快。
`+1` 是因为 `prefix` 是 1-based：`target` 取 `[1, 总权重]`。

- **复杂度**：初始化 O(n)；`pickIndex` O(log n)；空间 O(n)。
- **易错点**：`target` 的左右端点（1-based 取 `[1, total]`）与 `bisect_left` 的对应；
  权重是正整数，不必担心 0 段。C++ 用 `uniform` 时会引入 `<random>`，用 `rand() % total + 1`
  也行，但注意 `rand()` 在一段上分布略有不均，追求严格等概率时可换 `std::mt19937`。
- **相似题**：497. 非重叠矩形中的随机点（权重换成面积，套路相同）；本模式也是前缀和
  （第 04 篇）的一个「反用」：不再是求区间和，而是用前缀和定位。

### 497. 非重叠矩形中的随机点（中等）

**题目**：给定互不重叠的矩形 `rects[i] = [x1, y1, x2, y2]`（含边界整数格点），`pick()`
先按面积比例选矩形、再在矩形内均匀取整数点；等价于每个点被选中的概率正比于所在矩形面积。

**思路**：

```python
class Solution:
    def __init__(self, rects):
        self.rects = rects
        self.prefix = []
        total = 0
        for x1, y1, x2, y2 in rects:
            total += (x2 - x1 + 1) * (y2 - y1 + 1)
            self.prefix.append(total)

    def pick(self):
        target = random.randint(1, self.prefix[-1])
        i = bisect.bisect_left(self.prefix, target)
        x1, y1, x2, y2 = self.rects[i]
        return [random.randint(x1, x2), random.randint(y1, y2)]
```

```cpp
class Solution {
public:
    Solution(std::vector<std::vector<int>> rects) : rects_(rects) {
        int total = 0;
        for (const auto& r : rects_) {
            total += (r[2] - r[0] + 1) * (r[3] - r[1] + 1);
            prefix_.push_back(total);
        }
    }

    std::vector<int> pick() {
        int target = std::rand() % prefix_.back() + 1;
        int i = static_cast<int>(
            std::lower_bound(prefix_.begin(), prefix_.end(), target) -
            prefix_.begin());
        const auto& r = rects_[i];
        int x = r[0] + std::rand() % (r[2] - r[0] + 1);
        int y = r[1] + std::rand() % (r[3] - r[1] + 1);
        return {x, y};
    }

private:
    std::vector<std::vector<int>> rects_;
    std::vector<int> prefix_;
};
```

**为什么分两步仍然均匀**：矩形 i 的整数点数（即「面积」）是
`(x2 - x1 + 1) * (y2 - y1 + 1)`，把它当权重做前缀和，第一步就是 528 的加权抽样，选中矩形 i
的概率为 `它的点数 / 总点数`。第二步在矩形 i 内等概率取点，命中其中某个点再乘 1/它的点数，
约掉后每个点被选中的概率都是 `1 / 总点数`，与它在哪个矩形无关。

- **复杂度**：初始化 O(n)；`pick` O(log n)；空间 O(n)。
- **易错点**：点数要 `+1`（矩形含两端边界）；坐标取 `[x1, x2]`、`[y1, y2]` 的闭区间；
  二分定位的是矩形下标，不是在数轴上取点。题目保证矩形互不重叠，所以按面积切分没有歧义。
- **相似题**：528. 按权重随机选择（同一模板，权重=面积）；这两题与第 04 篇「前缀和与差分」
  的 303/304 是同一族结构，只是用途从「求区间和」变成「按区间长度定位」。

---

## 模式三：拒绝采样与黑名单映射

**适用信号**：等概率范围「很难直接构造」，或要避开一组非法取值（黑名单）。

**核心动作**：要么在一个**更大、易均匀**的范围里取样并丢弃非法结果（拒绝采样）；要么把
非法值**映射**到合法值上，让取值范围重新变成一段连续整数（黑名单映射）。

### 710. 黑名单中的随机数（困难）

**题目**：给定 n 和黑名单 `blacklist`（数字互异、都在 `[0, n)`），`pick()` 等概率返回
`[0, n)` 中一个不在黑名单里的整数。

**思路**：

```python
class Solution:
    def __init__(self, n, blacklist):
        m = len(blacklist)
        self.size = n - m
        blocked = set(blacklist)
        self.mapping = {}
        last = n - 1
        for b in blacklist:
            if b < self.size:
                while last in blocked:
                    last -= 1
                self.mapping[b] = last
                last -= 1

    def pick(self):
        idx = random.randint(0, self.size - 1)
        return self.mapping.get(idx, idx)
```

```cpp
class Solution {
public:
    Solution(int n, std::vector<int> blacklist) {
        int m = static_cast<int>(blacklist.size());
        size_ = n - m;
        std::unordered_set<int> blocked(blacklist.begin(), blacklist.end());
        int last = n - 1;
        for (int b : blacklist) {
            if (b < size_) {
                while (blocked.count(last)) {
                    --last;
                }
                mapping_[b] = last;
                --last;
            }
        }
    }

    int pick() {
        int idx = std::rand() % size_;
        auto it = mapping_.find(idx);
        return it == mapping_.end() ? idx : it->second;
    }

private:
    int size_;
    std::unordered_map<int, int> mapping_;
};
```

**为什么把黑名单「换到尾部」**：可用的数一共 `size = n - m` 个。我们只打算在 `[0, size)`
上等概率取整数；麻烦在于这段区间里可能混着黑名单数字。做法是从 `n-1` 往下扫，把落在
`[0, size)` 里的黑名单 `b` 映射到「尾部 `[size, n)` 中一个可用数」。这样区间 `[0, size)`
里的每个下标都恰好对应一个可用数：原本可用的保持原值，原本是黑名单的返回映射值。于是
均匀取下标就等于均匀取可用数。落在尾部 `[size, n)` 的黑名单不用管——我们根本不会去尾部取值。

- **复杂度**：初始化 O(m)；`pick` 平均 O(1)；空间 O(m)。
- **易错点**：只映射 `b < size` 的黑名单；`last` 要一路跳过所有黑名单（用集合判断），
  但不能把它降到 `size` 以下（可用数数量恰好够）；`pick` 用 `randint(0, size-1)`。
- **相似题**：528. 按权重随机选择（用映射把「不等概率」变成「等概率取下标」的同一思想）；
  若黑名单很大、n 很小时，也可直接枚举所有可用数存起来再等概率取，只是不够省空间。

### 470. 用 Rand7() 实现 Rand10()（中等）

**题目**：只有 `rand7()`（等概率返回 1..7），只用它实现 `rand10()`（等概率返回 1..10）。

**思路**：

```python
def rand7():
    return random.randint(1, 7)


def rand10():
    while True:
        r = (rand7() - 1) * 7 + rand7()
        if r <= 40:
            return (r - 1) % 10 + 1
```

```cpp
int rand7() {
    return std::rand() % 7 + 1;
}

int rand10() {
    while (true) {
        int r = (rand7() - 1) * 7 + rand7();
        if (r <= 40) {
            return (r - 1) % 10 + 1;
        }
    }
}
```

**为什么两次 rand7 就够**：一次只有 7 种结果，凑不出 10 的等概率。两次调用错开权重：
`(rand7() - 1) * 7 + rand7()` 得到 1..49 上的**均匀**分布（相当于七进制的两位数）。
40 是 10 的倍数，所以 1..40 均分给 1..10，每类恰好 4 个，`(r - 1) % 10 + 1` 就是等概率的。
落在 41..49 的 9 个值直接**拒绝**重抽，虽然多花几次调用，但保证结果严格均匀——若把它们
「折」回 1..9，就会出现某些结果概率偏高。

- **复杂度**：期望时间 O(1)（期望调用 `2 × 49/40 ≈ 2.45` 次 rand7），空间 O(1)。
- **易错点**：映射别写成 `rand7() * rand7()`（平方后不是均匀）；拒绝区间要成段、且是
  10 的倍数才好处理；`(rand7()-1)*7 + rand7()` 的偏移量是 7 不是 6。
- **相似题**：478. 在圆内随机生成点（同为拒绝采样）；710. 黑名单（不用拒绝而用映射），
  两种手法各有适用场景。

### 478. 在圆内随机生成点（中等）

**题目**：给定半径 `radius` 与圆心 `(x_center, y_center)`，等概率返回圆内（含边界）一点。

**思路**：

```python
class Solution:
    def __init__(self, radius, x_center, y_center):
        self.radius = radius
        self.x_center = x_center
        self.y_center = y_center

    def randPoint(self):
        while True:
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x * x + y * y <= 1:
                return [self.x_center + x * self.radius,
                        self.y_center + y * self.radius]
```

```cpp
class Solution {
public:
    Solution(double radius, double x_center, double y_center)
        : radius_(radius), x_center_(x_center), y_center_(y_center) {}

    std::vector<double> randPoint() {
        while (true) {
            double x = uniform();
            double y = uniform();
            if (x * x + y * y <= 1.0) {
                return {x_center_ + x * radius_, y_center_ + y * radius_};
            }
        }
    }

private:
    double uniform() {
        return 2.0 * std::rand() / RAND_MAX - 1.0;
    }

    double radius_;
    double x_center_;
    double y_center_;
};
```

**为什么在外接正方形里拒绝**：在外接正方形（边长 2R）里均匀取点很容易，落在圆内的点仍
保持「正方形内的均匀」——均匀分布被裁剪后还是均匀分布，只要把范围平移缩放到目标圆即可。
接受概率是圆的面积除以正方形面积 `πR² / (4R²) = π/4 ≈ 0.785`，期望重抽约 1.27 次。
如果改用极坐标，半径必须写成 `sqrt(U) · R`（U 均匀）才能保证面积均匀；直接取 `R·U` 会让点
向圆心聚集。拒绝采样不需要这个推导，更不容易出错。

- **复杂度**：期望时间 O(1)，空间 O(1)。
- **易错点**：坐标要在 `[-1, 1]` 上取（对应外接正方形），最后才乘 `radius` 平移；
  边界条件用 `<=`（含边界）。C++ 的 `RAND_MAX` 与整数除法要注意先转 `double`。
- **相似题**：470. 用 Rand7() 实现 Rand10()（拒绝采样）；随机点是「连续版拒绝采样」，
  rand10 是「离散版」。

---

## 模式四：水塘抽样

**适用信号**：数据只能**流式/单遍**读取（如链表），或不想存所有候选，但要求等概率取一个。

**核心动作**：扫描到第 i 个候选时，以 `1/i` 的概率把答案替换成它，否则保持不变。

### 382. 链表随机节点（中等）

**题目**：给单链表头结点，`getRandom()` 等概率返回某个节点的值；进阶要求只遍历一次、
额外空间 O(1)。

**思路**：

```python
class Solution:
    def __init__(self, head):
        self.head = head

    def getRandom(self):
        res = self.head.val
        node = self.head.next
        i = 2
        while node:
            if random.randint(1, i) == 1:
                res = node.val
            node = node.next
            i += 1
        return res
```

```cpp
class Solution {
public:
    Solution(ListNode* head) : head_(head) {}

    int getRandom() {
        int res = head_->val;
        ListNode* node = head_->next;
        int i = 2;
        while (node) {
            if (std::rand() % i == 0) {
                res = node->val;
            }
            node = node->next;
            ++i;
        }
        return res;
    }

private:
    ListNode* head_;
};
```

**为什么每个节点最终概率都是 1/n**：考察第 k 个节点留在答案里的概率。它要在第 k 步被选中
（概率 `1/k`），并且之后第 k+1…n 步都不能把它替换掉（第 i 步不被替换的概率是 `1 - 1/i`）：

```text
(1/k) · ∏_{i=k+1}^{n} (1 - 1/i) = (1/k) · ∏ (i-1)/i = (1/k) · (k/n) = 1/n
```

乘积一路「望远镜式」约掉，结果与 k 无关，正是等概率。这就是水塘抽样的核心证明。

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：第一个节点先无条件作为初始答案（`i` 从 2 开始），否则 `1/1` 的边界容易写错；
  本题的进阶要求是「只遍历一次、额外空间 O(1)」，所以不能先把链表存成数组再取随机。
- **相似题**：398. 随机数索引（数组版水塘抽样）。

### 398. 随机数索引（中等）

**题目**：数组可能含重复元素，`pick(target)` 等概率返回一个满足 `nums[i] == target` 的下标。

**思路**：

```python
class Solution:
    def __init__(self, nums):
        self.nums = nums

    def pick(self, target):
        res = -1
        count = 0
        for i, x in enumerate(self.nums):
            if x == target:
                count += 1
                if random.randint(1, count) == 1:
                    res = i
        return res
```

```cpp
class Solution {
public:
    Solution(std::vector<int> nums) : nums_(nums) {}

    int pick(int target) {
        int res = -1;
        int count = 0;
        for (int i = 0; i < static_cast<int>(nums_.size()); ++i) {
            if (nums_[i] == target) {
                ++count;
                if (std::rand() % count == 0) {
                    res = i;
                }
            }
        }
        return res;
    }

private:
    std::vector<int> nums_;
};
```

**为什么只数目标元素**：把「等于 target 的位置」当成一条子序列，`count` 就是它的长度。
每遇到一个就以 `1/count` 的概率替换答案，正是 382 的水塘抽样，所以每个目标下标被选中的
概率都是 `1/(target 出现次数)`。不需要事先把所有下标收集起来，额外空间 O(1)。

- **复杂度**：初始化 O(1)；`pick` 时间 O(n)、空间 O(1)。
- **易错点**：`count` 只在命中 target 时自增；`std::rand() % count == 0` 与
  `randint(1, count) == 1` 等价。若同一数组要被多次 `pick` 不同 target，可预存
  「值 → 下标列表」再用 528 的均匀取法，但空间换成了 O(n)。
- **相似题**：382. 链表随机节点（同一模板）；随机容器 381 若只做「随机取一个值」也可用
  水塘抽样，但 381 要求 O(1) 随机访问，才改用数组。

---

## 模式五：随机容器

**适用信号**：既要 O(1) 插入/删除，又要等概率随机取一个元素。

**核心动作**：**动态数组**负责「等概率取」（下标均匀），**哈希表**负责「O(1) 定位」；
删除时用**末尾元素填坑**，把「删除中间」变成「删除末尾」。

### 381. O(1) 时间插入、删除和获取随机元素 - 允许重复（困难）

**题目**：实现 `RandomizedCollection`：`insert(val)` 返回插入前是否不存在该值；
`remove(val)` 删除一个 val 并返回是否存在；`getRandom()` 按元素个数等概率返回一个元素。

**思路**：

```python
class RandomizedCollection:
    def __init__(self):
        self.nums = []
        self.pos = {}

    def insert(self, val):
        self.nums.append(val)
        self.pos.setdefault(val, set()).add(len(self.nums) - 1)
        return len(self.pos[val]) == 1

    def remove(self, val):
        if val not in self.pos:
            return False
        i = self.pos[val].pop()
        last = self.nums[-1]
        if i != len(self.nums) - 1:
            self.pos[last].discard(len(self.nums) - 1)
            self.pos[last].add(i)
        self.nums[i] = last
        self.nums.pop()
        if not self.pos[val]:
            del self.pos[val]
        return True

    def getRandom(self):
        return random.choice(self.nums)
```

```cpp
class RandomizedCollection {
public:
    RandomizedCollection() {}

    bool insert(int val) {
        nums_.push_back(val);
        pos_[val].insert(static_cast<int>(nums_.size()) - 1);
        return pos_[val].size() == 1;
    }

    bool remove(int val) {
        auto it = pos_.find(val);
        if (it == pos_.end()) {
            return false;
        }
        auto& idxs = it->second;
        int i = *idxs.begin();
        idxs.erase(idxs.begin());
        int lastIdx = static_cast<int>(nums_.size()) - 1;
        int last = nums_.back();
        if (i != lastIdx) {
            pos_[last].erase(lastIdx);
            pos_[last].insert(i);
        }
        nums_[i] = last;
        nums_.pop_back();
        if (idxs.empty()) {
            pos_.erase(it);
        }
        return true;
    }

    int getRandom() {
        return nums_[std::rand() % nums_.size()];
    }

private:
    std::vector<int> nums_;
    std::unordered_map<int, std::unordered_set<int>> pos_;
};
```

**为什么数组 + 下标集合**：`getRandom` 要等概率，数组下标天然均匀；但数组删除中间元素是
O(n)，而删除末尾是 O(1)。破局点是**顺序不重要**：删除 `val` 的某个下标 `i` 时，把数组末尾
元素搬到 `i` 处填坑，再弹出末尾即可。因为允许重复，同一个值会有多个下标，所以 `pos` 要从
380 的「值 → 单个下标」升级成「值 → 下标集合」。填坑会挪动末尾元素，所以必须同步把
`pos[last]` 里的「末尾下标」换成 `i`。

- **复杂度**：三个操作平均 O(1)；空间 O(n)。
- **易错点**：`remove` 里「迁移下标」和「写 `nums[i]`」的次序；`val` 恰好等于末尾元素时
  `pos[last]` 与 `pos[val]` 是同一个集合，先 `pop` 掉 `i` 再迁移不会出错；集合空了要删键。
- **相似题**：380. O(1) 时间插入、删除和获取随机元素（见第 19 篇「设计题」，不允许重复的
  版本）；519. 随机翻转矩阵（同样用哈希表维护「可用下标集合」）。

---

## 规律总结

1. **等概率是可以验证的，不要靠感觉**。写完后逐个候选算一遍「被选中的概率是否相同」：
   384 的 1/n!、水塘抽样的 1/n、710 的 1/可用数……算得通，才敢提交。批量跑样例、统计
   各值出现次数，是最实用的自检手段。

2. **洗牌只认 Fisher-Yates**：从后往前，第 i 位与 `[0, i]` 随机交换（含自己）。凡是要求
   「随机取一个未用过的元素」的题，都可以套这个骨架，用哈希表把「未用过集合」压缩成
   `[0, 剩余数)` 的一段连续下标（519、381）。

3. **加权抽样 = 前缀和 + 二分**。把权重（或面积、点数）摊成数轴，随机取点后二分定位。
   这是第 04 篇「前缀和」的反用：前缀和不再用来「求区间和」，而是用来「按区间长度定位」。

4. **拒绝采样：先扩大范围，再丢掉非法值**。范围越规整越好取（正方形、1..49）；接受概率
   不必是 1，但必须保证「合法结果内部仍然均匀」（470、478）。关键是不能把被拒绝的值「折」
   回去，否则概率会偏。

5. **映射 vs 拒绝**：黑名单 710 用**映射**把非法值换到合法尾部，一次命中、不重抽；
   470/478 用**拒绝**重抽。一个改「值域」、一个改「采样次数」，按题意选更省的那个。

6. **水塘抽样：一次遍历、O(1) 空间、以 `1/i` 替换**。适合数据流、链表、海量数据里「等概率
   取一个」。证明靠望远镜式连乘：`(1/k)·∏(1-1/i) = 1/n`，与 k 无关。

7. **随机容器：数组管「等概率取」，哈希表管「O(1) 定位」**。删除时用**末尾填坑**把中间
   删除降成末尾删除；允不允许重复，只差在哈希表里存「单个下标」还是「下标集合」。

8. **随机数不是越复杂越好**。`rand() % n` 在多数题里够用；若对均匀性要求严格，用
   `std::mt19937 + uniform_int_distribution`，并给 Python 的 `random.seed` 固定种子，
   让自测可复现。C++ 自测里把 `std::srand` 放在 `main`，别塞进类的构造函数。

9. **与其它篇的联系**：加权抽样用到第 04 篇的前缀和与第 05 篇的二分；随机容器 381 是
   第 19 篇「设计题」380 的延伸；「等概率 + 计数」的验证思路与第 16 篇「位运算」里
   「每个元素只算一次贡献」异曲同工。
