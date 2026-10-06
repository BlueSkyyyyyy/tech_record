# 队列与双端队列：模拟、设计与单调队列

队列是「先进先出」的容器：两个口，从尾进、从头出。它最擅长的事情是**忠实地
模拟「排队 / 轮流 / 按时间先后」的过程**。而双端队列（deque）两头都能进出，
于是又多出一种威力极大的用法——**单调队列**：在滑动窗口里 O(1) 地拿到极值。
本篇收 10 道题，按三种用法分组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：用队列模拟流程 | 1823. 找出游戏的获胜者 / 1700. 无法吃午餐的学生数量 | 简单 / 简单 |
| 模式二：双端队列的构造与设计 | 950. 按递增顺序显示卡牌 / 641. 设计循环双端队列 / 1670. 设计前中后队列 / 1352. 最后 K 个数的乘积 | 中等 ×4 |
| 模式三：单调队列（滑动窗口的极值） | 862. 和至少为 K 的最短子数组 / 1438. 绝对差不超过限制的最长连续子数组 / 1696. 跳跃游戏 VI / 1425. 带限制的子序列和 | 困难 / 中等 / 中等 / 困难 |

> 一句话记住三者的分工：**普通队列管「先后顺序」，双端队列管「两头操作」，
> 单调队列管「窗口极值」。** 单调队列是本篇的重点，它把「滑动窗口最值」从
> O(nk) 优化到 O(n)，是第 03 篇「滑动窗口」里 239 那道题的通用化。

---

## 模式一：用队列模拟流程

**适用信号**：题目描述了一个「一圈人 / 一排队列」按规则轮流进行处理的过程，
要求最终状态或最后的胜者。

**核心动作**：把参与者按顺序放进队列，队首就是「当前轮到的人」，处理完按规定
要么弹出、要么搬到队尾。队列的 FIFO 语义天然对应「处理完轮到下一位」。

### 1823. 找出游戏的获胜者（简单）

**题目**：n 个朋友围成一圈，编号 1..n。从 1 号开始报数，每报到 k 的人出局，
出局者的下一位从 1 重新报数。重复到只剩一人，返回获胜者编号。

**思路**：

```python
from collections import deque


def find_the_winner(n, k):
    q = deque(range(1, n + 1))
    while len(q) > 1:
        for _ in range(k - 1):
            q.append(q.popleft())
        q.popleft()
    return q[0]
```

```cpp
int findTheWinner(int n, int k) {
    std::queue<int> q;
    for (int i = 1; i <= n; ++i) {
        q.push(i);
    }
    while (q.size() > 1) {
        for (int i = 0; i < k - 1; ++i) {
            q.push(q.front());
            q.pop();
        }
        q.pop();
    }
    return q.front();
}
```

**为什么用队列**：报数这件事本来需要在数组里手动维护「当前下标」并处理绕圈取模，
很容易写错。放进队列后，「报到 1..k-1 的人安全」就变成「出队再入队」这个动作，
他们自动转到队尾等待下一轮；此时队首恰好就是报到 k 的人，弹掉即可。
每淘汰一人做 `k-1` 次搬移，共 n-1 轮，直观且不会错。

- **复杂度**：时间 O(n*k)，空间 O(n)。
- **易错点**：`k = 1` 时内层循环不执行，等价于每次删队首；只剩一人时返回
  `q[0]`（不是继续循环）。C++ 里 `std::queue::front()` 只读不删，删要用 `pop()`。
- **相似题**：约瑟夫环有 O(n) 的递推公式 `f(n,k) = (f(n-1,k) + k) % n`，
  不需要队列；这里的队列模拟胜在**过程清晰**。同类「轮流处理」还有 1700（下题）。

### 1700. 无法吃午餐的学生数量（简单）

**题目**：学生排队，sandwiches 是栈（`sandwiches[0]` 是栈顶）。队首学生若喜欢
栈顶的三明治就取走并离开，否则走到队尾。问到不能有人再取走时，还有多少学生吃不上。

**思路**：

```python
def count_students(students, sandwiches):
    zeros = students.count(0)
    ones = len(students) - zeros
    for s in sandwiches:
        if s == 0 and zeros > 0:
            zeros -= 1
        elif s == 1 and ones > 0:
            ones -= 1
        else:
            break
    return zeros + ones
```

```cpp
int countStudents(std::vector<int>& students, std::vector<int>& sandwiches) {
    int zeros = 0;
    for (int s : students) {
        if (s == 0) {
            ++zeros;
        }
    }
    int ones = static_cast<int>(students.size()) - zeros;
    for (int s : sandwiches) {
        if (s == 0 && zeros > 0) {
            --zeros;
        } else if (s == 1 && ones > 0) {
            --ones;
        } else {
            break;
        }
    }
    return zeros + ones;
}
```

**为什么可以省掉队列**：真用队列一轮轮转也能做，但要想清楚一个事实——学生被拒绝
之后只是回到队尾，**他们的相对顺序对结果毫无影响**，真正决定成败的只有「喜欢 0
和喜欢 1 各剩多少人」。于是只要按栈顶顺序发三明治，栈顶那种没人要了就停下，
剩下的人数就是答案。这是在模拟题里常见的简化：先问「过程里真正起作用的量是什么」。

- **复杂度**：时间 O(n + m)，空间 O(1)。
- **易错点**：停下的条件是「栈顶那种口味的学生已经为 0」，不能只判断当前队首；
  初始就要分别数出 0 和 1 的人数。
- **相似题**：1823（上题）同样是「队列模拟」，但顺序重要，所以不能这样化简。

---

## 模式二：双端队列的构造与设计

**适用信号**：题目要在**两头**插入/删除，或者要用「从结果倒推初始」的方式构造
一个序列，又或者要在 O(1) 时间里存取「前中后」多个端点。

**核心动作**：用双端队列让两端的操作都变成 O(1)；环形数组则用 `head + count`
两个量精确描述「队列占了环的哪一段」。

### 950. 按递增顺序显示卡牌（中等）

**题目**：有一副牌，翻牌规则是「亮出牌堆顶、然后把下一张移到牌堆底、重复」。
已知最终亮出的顺序是递增的，求牌堆一开始的排列。

**思路**：

```python
from collections import deque


def deck_revealed_increasing(deck):
    d = deque()
    for card in sorted(deck, reverse=True):
        if d:
            d.appendleft(d.pop())
        d.appendleft(card)
    return list(d)
```

```cpp
std::vector<int> deckRevealedIncreasing(std::vector<int>& deck) {
    std::sort(deck.begin(), deck.end());
    std::deque<int> d;
    for (int i = static_cast<int>(deck.size()) - 1; i >= 0; --i) {
        if (!d.empty()) {
            d.push_front(d.back());
            d.pop_back();
        }
        d.push_front(deck[i]);
    }
    return std::vector<int>(d.begin(), d.end());
}
```

**为什么要倒着构造**：正过程是「弹出一张、把下一张搬到底」，很难从结果直接推初始。
但我们知道亮出的序列就是排好序的牌。于是从大到小把每张牌插到牌堆**最前面**；
插之前，如果堆里已有牌，先把堆底那张搬到堆顶——这正是正过程「把顶搬到底」的逆操作。
按牌面从大到小插入，保证每次被搬到底的，恰好是下一次会被亮出的牌。

- **复杂度**：时间 O(n log n)（排序），空间 O(n)。
- **易错点**：顺序是「先搬堆底到堆顶，再插新牌」，两步不能颠倒；第一张牌入堆前
  没有「搬底」动作，要加 `if` 判断。Python 的 `deque` 用 `appendleft` / `pop`，
  C++ 用 `push_front` / `pop_back` / `back`。
- **相似题**：641、1670（下两题）都用双端队列做两头操作；本题是「倒序构造」。

### 641. 设计循环双端队列（中等）

**题目**：实现一个固定容量 k 的循环双端队列，支持 `insertFront` / `insertLast` /
`deleteFront` / `deleteLast` / `getFront` / `getRear` / `isEmpty` / `isFull`。

**思路**：

```python
class MyCircularDeque:
    def __init__(self, k):
        self.cap = k
        self.data = [0] * k
        self.head = 0
        self.count = 0

    def insertFront(self, value):
        if self.isFull():
            return False
        self.head = (self.head - 1) % self.cap
        self.data[self.head] = value
        self.count += 1
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        self.data[(self.head + self.count) % self.cap] = value
        self.count += 1
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.cap
        self.count -= 1
        return True

    def deleteLast(self):
        if self.isEmpty():
            return False
        self.count -= 1
        return True

    def getFront(self):
        if self.isEmpty():
            return -1
        return self.data[self.head]

    def getRear(self):
        if self.isEmpty():
            return -1
        return self.data[(self.head + self.count - 1) % self.cap]

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.cap
```

```cpp
class MyCircularDeque {
public:
    explicit MyCircularDeque(int k) : cap_(k), data_(k, 0), head_(0), count_(0) {}

    bool insertFront(int value) {
        if (isFull()) {
            return false;
        }
        head_ = (head_ - 1 + cap_) % cap_;
        data_[head_] = value;
        ++count_;
        return true;
    }

    bool insertLast(int value) {
        if (isFull()) {
            return false;
        }
        data_[(head_ + count_) % cap_] = value;
        ++count_;
        return true;
    }

    bool deleteFront() {
        if (isEmpty()) {
            return false;
        }
        head_ = (head_ + 1) % cap_;
        --count_;
        return true;
    }

    bool deleteLast() {
        if (isEmpty()) {
            return false;
        }
        --count_;
        return true;
    }

    int getFront() const {
        if (isEmpty()) {
            return -1;
        }
        return data_[head_];
    }

    int getRear() const {
        if (isEmpty()) {
            return -1;
        }
        return data_[(head_ + count_ - 1) % cap_];
    }

    bool isEmpty() const { return count_ == 0; }
    bool isFull() const { return count_ == cap_; }

private:
    int cap_;
    std::vector<int> data_;
    int head_;
    int count_;
};
```

**为什么用 `head + count`**：环形数组里如果只存「头指针」和「尾指针」，空队列和
满队列会表现成同一种状态（头碰尾），必须浪费一个格子区分。改用 `count` 记录元素
个数后，空就是 `count == 0`、满就是 `count == cap`，干净利落。队首下标是 `head`，
队尾下标是 `(head + count - 1) % cap`，所有移动都套一层 `% cap` 就实现了「循环」。

- **复杂度**：所有操作 O(1)，空间 O(k)。
- **易错点**：头部插入要 `-1` 后取模，C++ 里必须写成 `(head - 1 + cap) % cap`，
  否则负数取模会得到负下标；`deleteLast` 只需 `count--`，不用动 `head`。
- **相似题**：第 19 篇「设计题」的 622 设计循环队列用同一套 `head + count` 思路
  （只能一头进、一头出）；1670（下题）用两个双端队列实现「前中后」都能操作。

### 1670. 设计前中后队列（中等）

**题目**：实现一个队列，支持 `pushFront` / `pushMiddle` / `pushBack` /
`popFront` / `popMiddle` / `popBack`。元素个数为偶数时，「中间」取靠前的那个。

**思路**：

```python
from collections import deque


class FrontMiddleBackQueue:
    def __init__(self):
        self.a = deque()
        self.b = deque()

    def pushFront(self, val):
        self.a.appendleft(val)
        if len(self.a) > len(self.b) + 1:
            self.b.appendleft(self.a.pop())

    def pushMiddle(self, val):
        if len(self.a) > len(self.b):
            self.b.appendleft(self.a.pop())
        self.a.append(val)

    def pushBack(self, val):
        self.b.append(val)
        if len(self.b) > len(self.a):
            self.a.append(self.b.popleft())

    def popFront(self):
        if not self.a:
            return -1
        val = self.a.popleft()
        if len(self.a) < len(self.b):
            self.a.append(self.b.popleft())
        return val

    def popMiddle(self):
        if not self.a:
            return -1
        val = self.a.pop()
        if len(self.a) < len(self.b):
            self.a.append(self.b.popleft())
        return val

    def popBack(self):
        if not self.b:
            return self.a.pop() if self.a else -1
        val = self.b.pop()
        if len(self.a) > len(self.b) + 1:
            self.b.appendleft(self.a.pop())
        return val
```

```cpp
class FrontMiddleBackQueue {
public:
    void pushFront(int val) {
        a_.push_front(val);
        if (static_cast<int>(a_.size()) > static_cast<int>(b_.size()) + 1) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
    }

    void pushMiddle(int val) {
        if (a_.size() > b_.size()) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
        a_.push_back(val);
    }

    void pushBack(int val) {
        b_.push_back(val);
        if (b_.size() > a_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
    }

    int popFront() {
        if (a_.empty()) {
            return -1;
        }
        int val = a_.front();
        a_.pop_front();
        if (a_.size() < b_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
        return val;
    }

    int popMiddle() {
        if (a_.empty()) {
            return -1;
        }
        int val = a_.back();
        a_.pop_back();
        if (a_.size() < b_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
        return val;
    }

    int popBack() {
        if (b_.empty()) {
            if (a_.empty()) {
                return -1;
            }
            int val = a_.back();
            a_.pop_back();
            return val;
        }
        int val = b_.back();
        b_.pop_back();
        if (static_cast<int>(a_.size()) > static_cast<int>(b_.size()) + 1) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
        return val;
    }

private:
    std::deque<int> a_;
    std::deque<int> b_;
};
```

**为什么是两个队列**：一个队列没法同时 O(1) 地操作头和中间。用双端队列 `a` 存
前半段（含中间）、`b` 存后半段，并维持不变量 **`len(a) == len(b)` 或
`len(a) == len(b) + 1`**。这样 `a` 的队尾 `a[-1]` 永远就是中间元素，`b` 的队尾
就是队尾元素。每次操作后只需搬动**一个**元素即可恢复不变量，所以全是 O(1)。

- **复杂度**：所有操作 O(1)，空间 O(n)。
- **易错点**：`popBack` 要特判 `b` 为空（此时全部元素都在 `a`，且只有一个）；
  `pushMiddle` 在 `a` 比 `b` 多时，要先把 `a` 队尾挪到 `b` 队首再放入新元素，
  否则中间位置会算错。偶数个时中间取「靠前」的那个，靠 `len(a) == len(b)` 时
  中间落在 `a[-1]` 来保证。
- **相似题**：641（上题）是环形数组版的双端队列；本题是「双队列分担前后两半」，
  与第 08 篇「堆」里 295 中位数用的「对顶堆」是同一招思想（左右结构分担）。

### 1352. 最后 K 个数的乘积（中等）

**题目**：实现一个数据结构，支持 `add(num)` 往末尾追加一个数（num >= 0），
以及 `getProduct(k)` 返回最后 k 个数的乘积。

**思路**：

```python
class ProductOfNumbers:
    def __init__(self):
        self.prefix = [1]

    def add(self, num):
        if num == 0:
            self.prefix = [1]
        else:
            self.prefix.append(self.prefix[-1] * num)

    def getProduct(self, k):
        if k >= len(self.prefix):
            return 0
        return self.prefix[-1] // self.prefix[-1 - k]
```

```cpp
class ProductOfNumbers {
public:
    ProductOfNumbers() { prefix_.push_back(1); }

    void add(int num) {
        if (num == 0) {
            prefix_.clear();
            prefix_.push_back(1);
        } else {
            prefix_.push_back(prefix_.back() * static_cast<long long>(num));
        }
    }

    int getProduct(int k) {
        if (k >= static_cast<int>(prefix_.size())) {
            return 0;
        }
        return static_cast<int>(prefix_.back() / prefix_[prefix_.size() - 1 - k]);
    }

private:
    std::vector<long long> prefix_;
};
```

**为什么遇 0 就重置**：区间积本来可以用「前缀积相除」得到，可一旦出现 0，
前缀积会永远卡在 0，除法也失效。于是约定：**遇到 0 就把前缀积数组清空重置为
`[1]`**，等价于「历史从现在重新开始」。`prefix[-1]` 就是「自上次 0 以来所有数的
积」；`getProduct(k)` 如果 `k` 超过了这段历史的长度（`k >= len(prefix)`），说明窗口
伸进了那次 0，答案必为 0；否则 `prefix[-1] // prefix[-1 - k]` 就是区间积。
这是「用前缀结构 + 一个重置边界处理 0」的典型套路。

- **复杂度**：`add` O(1)，`getProduct` O(1)，空间 O(非零元素个数)。
- **易错点**：判断是 `k >= len(prefix)`（不是 `>`），因为 `prefix` 长度是
  「非零个数 + 1」；C++ 用 `long long` 存前缀积防溢出（题目保证答案在 32 位内）。
- **相似题**：第 04 篇「前缀和与差分」里 238 除自身以外数组的乘积、1524 等
  同属前缀积/前缀和家族；本题的「遇 0 重置」是这些题没涉及的新细节。

---

## 模式三：单调队列（滑动窗口的极值）

**适用信号**：DP 转移里要取「长度为 k 的窗口内最大值/最小值」，或者要在滑动窗口
里实时知道极值，而 k 可能很大。

**核心动作**：用双端队列维护一个**单调**的下标序列（存下标，比较靠下标取值）。
- 队首是窗口极值；
- 入队前从队尾弹掉「不可能再更优」的旧值，保持单调；
- 计算前从队首弹掉滑出窗口的下标。

关键认识：**一个「又旧又不优」的元素是废物**——它比新元素更早过期，值还不如新元素，
将来绝不会被选中，可以直接扔。

### 862. 和至少为 K 的最短子数组（困难）

**题目**：给定整数数组 nums（可能有负数）和整数 k，返回元素和 >= k 的最短非空
子数组的长度；不存在返回 -1。

**思路**：

```python
from collections import deque


def shortest_subarray(nums, k):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    ans = n + 1
    dq = deque()
    for j in range(n + 1):
        while dq and prefix[j] - prefix[dq[0]] >= k:
            ans = min(ans, j - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[j]:
            dq.pop()
        dq.append(j)
    return ans if ans <= n else -1
```

```cpp
int shortestSubarray(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + nums[i];
    }

    int ans = n + 1;
    std::deque<int> dq;
    for (int j = 0; j <= n; ++j) {
        while (!dq.empty() && prefix[j] - prefix[dq.front()] >= k) {
            ans = std::min(ans, j - dq.front());
            dq.pop_front();
        }
        while (!dq.empty() && prefix[dq.back()] >= prefix[j]) {
            dq.pop_back();
        }
        dq.push_back(j);
    }
    return ans <= n ? ans : -1;
}
```

**为什么普通双指针不行**：子数组和是「前缀和之差」。如果数组全是正数，前缀和
单调递增，双指针「右端越走和越大」成立；但这里有负数，前缀和会上下起伏，
双指针的单调前提被破坏。改用单调队列维护一组「可能当左端」的下标：

- 对每个右端 `j`，只要 `prefix[j] - prefix[队首] >= k` 就更新答案并弹出队首——
  队首更靠左，长度更长，对「最短」已无价值，而且后面不会有更靠左的队首了；
- 入队 `j` 前，把队尾 `prefix` 值 >= `prefix[j]` 的都弹掉：它们更旧（下标更小，
  更容易过期）且前缀和更大（作左端更差），注定没用。
队列里的前缀和因此严格递增。每个下标进出各一次，总 O(n)。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：子数组和对应 `prefix[j] - prefix[i]`，长度是 `j - i`（不是 `j-i+1`），
  因为 `i` 是前缀下标；第一层弹出要用 `while`（一个 `j` 可能连满足多个左端）；
  `k` 可能很大，C++ 前缀和用 `long long`。
- **相似题**：第 03 篇「滑动窗口」的 239 滑动窗口最大值是单调队列在定长窗口上的
  基础版；本题多了「前缀和 + 负数的非单调性」这层。209 长度最小的子数组是它
  在全正数下的简化版（用普通滑动窗口即可）。

### 1438. 绝对差不超过限制的最长连续子数组（中等）

**题目**：给整数数组 nums 和 limit，返回最长的连续子数组，使其中任意两元素之差
的绝对值都不超过 limit。

**思路**：

```python
from collections import deque


def longest_subarray(nums, limit):
    maxq = deque()
    minq = deque()
    left = 0
    ans = 0
    for right, x in enumerate(nums):
        while maxq and maxq[-1] < x:
            maxq.pop()
        maxq.append(x)
        while minq and minq[-1] > x:
            minq.pop()
        minq.append(x)

        while maxq[0] - minq[0] > limit:
            if maxq[0] == nums[left]:
                maxq.popleft()
            if minq[0] == nums[left]:
                minq.popleft()
            left += 1

        ans = max(ans, right - left + 1)
    return ans
```

```cpp
int longestSubarray(std::vector<int>& nums, int limit) {
    std::deque<int> maxq, minq;
    int left = 0, ans = 0;
    int n = static_cast<int>(nums.size());
    for (int right = 0; right < n; ++right) {
        int x = nums[right];
        while (!maxq.empty() && maxq.back() < x) {
            maxq.pop_back();
        }
        maxq.push_back(x);
        while (!minq.empty() && minq.back() > x) {
            minq.pop_back();
        }
        minq.push_back(x);

        while (maxq.front() - minq.front() > limit) {
            if (maxq.front() == nums[left]) {
                maxq.pop_front();
            }
            if (minq.front() == nums[left]) {
                minq.pop_front();
            }
            ++left;
        }
        ans = std::max(ans, right - left + 1);
    }
    return ans;
}
```

**为什么是「最大 - 最小」**：「窗口内任意两元素之差 <= limit」这个条件，等价于
「窗口最大值 - 窗口最小值 <= limit」——差最大的两元素一定是一头一尾的极值。
于是用两条单调队列同时维护窗口的最大值和最小值：`maxq` 从队首到队尾递减，
`minq` 递增，队首分别是最大、最小。右端每扩一格就压入并维护单调；若极值差超限，
左端右移收缩，移动时把「正好等于某队首」的左端元素从队首弹出。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：收缩左端时，只有当左端元素**恰好等于队首**才需要弹出（否则它早已
  被从队尾挤掉、不在队列里）；两条队列的队首可能同时等于左端元素，要分别判断。
- **相似题**：第 03 篇「滑动窗口」的 239 只维护最大值；1004 最大连续 1 的个数 III
  是「可容忍杂质」的变长窗。本题是「双单调队列 + 变长窗」的组合。

### 1696. 跳跃游戏 VI（中等）

**题目**：从下标 0 出发，每次最多向右跳 k 步，落在 `nums[i]` 就把 `nums[i]`
加进得分。问到达最后一个下标能得到的最大得分。

**思路**：

```python
from collections import deque


def max_result(nums, k):
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])
    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()
        dp[i] = nums[i] + dp[dq[0]]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return dp[-1]
```

```cpp
int maxResult(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    dp[0] = nums[0];
    std::deque<int> dq;
    dq.push_back(0);
    for (int i = 1; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) {
            dq.pop_front();
        }
        dp[i] = nums[i] + dp[dq.front()];
        while (!dq.empty() && dp[dq.back()] <= dp[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
    }
    return dp[n - 1];
}
```

**为什么单调队列能优化 DP**：状态转移是
`dp[i] = nums[i] + max(dp[j])`，j 取 `i-k .. i-1`。这正好是「长度 k 的滑动窗口
取最大值」。用单调队列在线维护这个窗口：先弹掉滑出的队首，队首即窗口最大 dp；
算出 `dp[i]` 后，从队尾弹掉 dp 不大于它的下标（又旧又不优），再入队。
于是每次转移 O(1)，总复杂度从 O(nk) 降到 O(n)。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：队列里存的是**下标**，比较和取值都要先 `dp[下标]`；弹队首的条件是
  `dq[0] < i - k`（下标超出窗口），不是大于；`dp[0]` 要单独初始化。
- **相似题**：第 13 篇「动态规划」的 198 打家劫舍、45 跳跃游戏 II 是同类跳步/取
  邻居的 DP；1425（下题）是同一模板换一个转移式。239 是纯滑动窗口最大值。

### 1425. 带限制的子序列和（困难）

**题目**：给整数数组 nums 和整数 k，求一个非空子序列的最大和，要求子序列中相邻
两个元素在原数组里的下标之差不超过 k。

**思路**：

```python
from collections import deque


def constrained_subset_sum(nums, k):
    n = len(nums)
    dp = [0] * n
    dq = deque()
    ans = nums[0]
    for i in range(n):
        while dq and dq[0] < i - k:
            dq.popleft()
        best = max(0, dp[dq[0]]) if dq else 0
        dp[i] = nums[i] + best
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        ans = max(ans, dp[i])
    return ans
```

```cpp
int constrainedSubsetSum(std::vector<int>& nums, int k) {
    int n = static_cast<int>(nums.size());
    std::vector<int> dp(n, 0);
    std::deque<int> dq;
    int ans = nums[0];
    for (int i = 0; i < n; ++i) {
        while (!dq.empty() && dq.front() < i - k) {
            dq.pop_front();
        }
        int best = dq.empty() ? 0 : std::max(0, dp[dq.front()]);
        dp[i] = nums[i] + best;
        while (!dq.empty() && dp[dq.back()] <= dp[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
        ans = std::max(ans, dp[i]);
    }
    return ans;
}
```

**和 1696 差在哪**：状态定义同样是「以 `i` 结尾的最大和」，转移同样是取窗口内
dp 的最大值，但本题多了一个选择——**可以断开、从自己重新开始**，所以写成
`dp[i] = nums[i] + max(0, max(dp[j]))`。前面那段和为正才值得接上，为负不如另起。
此外答案不一定落在最后一个下标，所以要**边算边记录全局最大 `ans`**。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：`max(0, ...)` 里的 0 表示「前面都不接」，漏了会把负数前缀强行接上；
  答案要取所有 `dp[i]` 的最大值；队列为空时 `best` 取 0。
- **相似题**：1696（上题）是「必须从起点连续跳到终点」，本题是「可断开的子序列」；
  第 13 篇「动态规划」的 53 最大子数组和是本模板在「窗口无限大」时的退化。

---

## 规律总结

1. **队列的核心语义是「先后顺序」**。1823 的报数、1700 的排队，都把「轮到谁」
   交给队首，把「处理完轮到下一位」交给「出队再入队」。能这样建模的题，
   不要去手工维护下标绕圈。

2. **模拟题先问「过程里真正起作用的量」**。1700 表面要转队列，实际只需两种口味
   的剩余人数；1823 表面要绕圈，队列模拟最直观。判断标准是：**顺序会不会影响
   结果**——会，就得真模拟；不会，就可以化简成计数。

3. **双端队列的价值在「两头 O(1)」**。950 用「插队首 + 搬堆底到堆顶」倒着构造，
   641 / 1670 用它实现两端的增删。环形数组记住公式：`head` 是队首，`count` 定
   空满，队尾 `(head + count - 1) % cap`。

4. **空满判定优先用 `count`**。`head == tail` 区分不了空和满，`head + count`
   则一目了然。这和第 19 篇「设计题」的 622 循环队列是同一个套路。

5. **「对顶结构」分担两端**。1670 用前后两个双端队列、295 用左右两个堆，都是
   「把整体拆成两半、维持大小平衡、只在边界处搬一个元素」的思想，目标都是 O(1)。

6. **单调队列存下标，比较靠取值**。队首是极值，入队前从队尾弹掉「又旧又不优」
   的元素，计算前从队首弹掉过期元素。这两步是固定模板，862 / 1438 / 1696 / 1425
   都只差比较的对象。

7. **判断「谁该被弹」就一句话：更早期 + 更差 = 废物**。下标更小意味着先过期，
   值更差意味着将来也不会被选，两者同时成立就可以直接扔。记住这句话，
   就不会在「是 `<` 还是 `<=`」上纠结太久（这里用 `<=` 也更省）。

8. **单调队列是滑动窗口最值的通用加速器**。239 是基础版；862 把它装到前缀和上
   解决「负数破坏单调性」；1696 / 1425 把它装到 DP 转移上，把取窗口最值的
   O(k) 降到 O(1)，整体 O(nk) → O(n)。

9. **前缀结构遇到 0 / 负数要断链或换工具**。1352 遇到 0 就重置前缀积；862 遇到
   负数就不能用双指针。前者用「重置边界」，后者用「单调队列」，都是为「破坏
   单调性的元素」准备的后路。

10. **与其它篇的联系**：单调队列与第 07 篇「栈与单调栈」的单调栈是一对孪生
    工具——栈处理「下一个更大元素」这类**边界**问题，队列处理「窗口内最值」
    这类**范围**问题；641 / 1670 / 1352 的容器设计与第 19 篇「设计题」同源；
    1696 / 1425 的「窗口最值优化 DP」属于第 13 篇「动态规划」的常用加速手段。
