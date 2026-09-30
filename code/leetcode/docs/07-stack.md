# 栈

栈是另一种「线性表」，但规矩比数组和链表都严格：**只能在一端进出，后进先出（LIFO）**。
正因为它这么「死板」，反而特别适合处理一类特定问题——**需要“回到最近一次未处理的状态”**。
生活里最直观的例子是浏览器的后退、编辑器的撤销：最新的那一步总是最先被撤销。

栈的价值不在存储，而在「顺序」。当问题的处理顺序天然是「后来的先处理」时，栈往往能让代码变得极短：
括号匹配、表达式求值、单调栈找下一个更大元素，都源于这一点。本篇先用四道基础题把栈
「怎么存、怎么用、怎么和别的结构互转」讲清楚，G-2 再进入更有技巧的单调栈。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 匹配与消消看 | 20. 有效的括号 | 简单 |
| 辅助栈记录极值 | 155. 最小栈 | 简单 |
| 受限容器互相模拟 | 232. 用栈实现队列 | 简单 |
| 受限容器互相模拟 | 225. 用队列实现栈 | 简单 |

---

## 模式一：匹配与消消看

**适用信号**：问题是「两个东西是否成对」「顺序是否正确」，且配对要求**后来者先配对**。
关键词是「括号」「配对」「嵌套」「消去」。

核心动作：用一个栈存放「已经出现、但还没被配对的左半边」。
遇到右半边时，只检查栈顶——栈顶就是「最近一个等待配对的左半边」，
配得上就弹出（这一对消掉），配不上就说明顺序错误。

### 20. 有效的括号（简单）

**题目**：给定一个只包括 `'('`、`')'`、`'['`、`']'`、`'{'`、`'}'` 的字符串 `s`，判断字符串是否有效。有效需满足：左括号必须用相同类型的右括号闭合，且必须以正确的顺序闭合。

**思路（用栈记住最近未闭合的左括号）**：
从左到右扫描字符串。遇到左括号就压栈；遇到右括号，就去看栈顶是不是与它配对的左括号：
是，就弹掉栈顶（这一对闭合了）；不是（或栈为空），说明顺序错了，直接返回 `False`。
扫描结束后，栈必须为空，才说明所有左括号都被闭合了。

为什么用栈：括号匹配的规则是「后出现的左括号必须先被闭合」。每遇到一个右括号，
要配对的永远是「最近一个还没闭合的左括号」，而栈顶恰好就是它。这个「最近优先」正是 LIFO 的语义。

为什么扫描结束还要判空栈：像 `"((("` 这种只有左括号的串，扫描过程不会出错，
但确实存在未闭合的括号，所以必须靠「最后栈为空」兜住。反过来 `")"` 在遇到右括号时栈已空，当场就返回了 `False`。

**代码**（完整可运行版见 `src/stack/valid_parentheses.py` / `.cpp`）：

```python
def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack
```

```cpp
bool isValid(const std::string &s) {
    std::stack<char> st;
    for (char ch : s) {
        if (ch == '(' || ch == '[' || ch == '{') {
            st.push(ch);
        } else {
            if (st.empty()) return false;
            char top = st.top();
            st.pop();
            if ((ch == ')' && top != '(') ||
                (ch == ']' && top != '[') ||
                (ch == '}' && top != '{')) {
                return false;
            }
        }
    }
    return st.empty();
}
```

- **复杂度**：时间 O(n)（每个字符进出栈各一次），空间 O(n)（最坏情况全是左括号）。
- **易错点**：遇到右括号时必须先判「栈是否为空」，否则取栈顶会越界；右括号与栈顶不匹配要立刻返回 `False`，不能只是不弹出而继续；最后返回的是 `not stack`，漏掉这一步会把 `"("` 判成有效；空字符串是有效的，无需特判（`not stack` 自然为 `True`）。
- **相似题**：394. 字符串解码（用栈保存「外层字符串 + 重复次数」，遇到 `[` 入栈、`]` 出栈拼接，见 `stack` 续篇）；1249. 移除无效的括号（同样用栈记录待配对位置）；32. 最长有效括号（在匹配的基础上用栈存下标求最长段）。匹配类的通用套路都是「用栈保存未配对的左半边」。

---

## 模式二：辅助栈记录极值

**适用信号**：普通容器已经在正常工作，但题目额外要求**在 O(1) 内查到某个全局性质**
（最小值、最大值），而这个性质会随插入删除而变。关键词是「常数时间取最小/最大」「设计」。
核心动作：**再开一个辅助结构，让它在每一步都同步保存“当前所求的极值”**，
把计算成本从查询时前移到更新时。

### 155. 最小栈（简单）

**题目**：设计一个支持 `push`、`pop`、`top` 操作，并能在常数时间内检索到最小元素的栈。需要实现 `MinStack` 类：`push(val)` 压入元素；`pop()` 删除栈顶元素；`top()` 获取栈顶元素；`get_min()` 检索栈中最小元素。

**思路（让最小栈与主栈同步）**：
用一个主栈存所有元素，再开一个「最小栈」，让它跟着主栈一起长、一起缩：
主栈每压入一个 `val`，若 `val` 不大于最小栈栈顶，就把 `val` 也压入最小栈；
主栈每弹出一个 `val`，若它恰好等于最小栈栈顶，就同步弹出最小栈栈顶。
这样最小栈的栈顶永远等于当前主栈里的最小值，`get_min` 直接取栈顶即可。

为什么最小栈要「压入不大于栈顶的值」，而不是「只记录严格更小的值」：
因为要处理重复的最小值。假设栈里先后压入两个相同的 `5`，且都是当前最小。
若最小栈只在严格更小时才压入，那么第一个 `5` 进入后，弹出它时最小栈顶就提前变掉了，
第二个 `5` 就被漏掉，`get_min` 会给出错误答案。加上等号后，每个最小值都在最小栈里有对应的一份，
弹出时也能一一对应地弹。

为什么 `get_min` 能做到 O(1)：最小值始终被缓存在最小栈的栈顶，查询时不需要现场遍历，
成本被前移到了每次 `push`/`pop` 里。

**代码**（`src/stack/min_stack.py` / `.cpp`）：

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]
```

```cpp
class MinStack {
  public:
    void push(int val) {
        stack_.push(val);
        if (minStack_.empty() || val <= minStack_.top()) minStack_.push(val);
    }

    void pop() {
        int val = stack_.top();
        stack_.pop();
        if (val == minStack_.top()) minStack_.pop();
    }

    int top() { return stack_.top(); }

    int getMin() { return minStack_.top(); }

  private:
    std::stack<int> stack_;
    std::stack<int> minStack_;
};
```

- **复杂度**：各操作时间 O(1)，空间 O(n)（最坏情况最小栈与主栈等长，比如元素单调不增）。
- **易错点**：压入判据用 `<=` 而不是 `<`，否则遇到重复最小值会出错；弹出时要比较的是「弹出值是否等于最小栈顶」，相等才弹，不等说明它不是当前最小值、最小栈不动；`pop`/`top` 调用前题目保证栈非空，无需额外判空。
- **相似题**：716. 最大栈（镜像问题，O(1) 取最大值，同时支持弹出最大值）；239. 滑动窗口最大值（同样维护「窗口内的极值」，但用单调队列而非辅助栈，见 `sliding-window`）；150. 逆波兰表达式求值（主栈存中间结果，是栈的另一种基础用法）。

---

## 模式三：受限容器互相模拟

**适用信号**：题目让你**只准用一种容器去实现另一种容器的语义**。这类题考的是对两种结构
「进出规则」的理解，而不是算法。关键词是「用栈实现队列」「用队列实现栈」「仅使用」。
核心动作：**利用“反转两次等于没反转”或“旋转队列”的手法，把 LIFO 与 FIFO 互相翻译**。

### 232. 用栈实现队列（简单）

**题目**：请你仅使用两个栈实现先入先出队列，支持 `push(x)`（把元素推到队列末尾）、`pop()`（从队列开头移除并返回元素）、`peek()`（返回队列开头元素）、`empty()`（队列为空返回 `True`）。

**思路（两个栈分工：一个只管进、一个只管出）**：
用 `in_stack` 接收 `push`，用 `out_stack` 负责弹出。
`push` 时直接压入 `in_stack`；需要 `pop`/`peek` 时，如果 `out_stack` 为空，
就把 `in_stack` 里的元素**全部倒进** `out_stack`，然后从 `out_stack` 取栈顶。

为什么这样能得到队列顺序：一个栈会把顺序反转一次，两个栈就把顺序反转两次、变回原序。
仔细看倒的过程：最先进入 `in_stack` 的元素沉在栈底，倒的时候它最后被弹出、最先被压入
`out_stack`，于是跑到 `out_stack` 的栈顶，正好是要先出的队头；而最晚进来的元素在
`in_stack` 顶部，最先被弹、最后被压，落在 `out_stack` 底部，最后才出。
两次「后进先出」叠加，得到的就是先进先出。

为什么「只在 `out_stack` 为空时才倒」：每次 `push` 后如果都重新倒一遍，
会把原本已经排好序的元素重新搅在一起，还会做无谓的搬运。`out_stack` 里还有元素时，
它的栈顶一定比 `in_stack` 里任何元素都更早入队，直接取即可。

为什么整体是摊还 O(1)：每个元素最多从 `in_stack` 搬到 `out_stack` 一次，
n 次操作里总搬运次数不超过 n，均摊到每次操作就是常数。

**代码**（`src/stack/implement_queue_using_stacks.py` / `.cpp`）：

```python
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self.peek()
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack
```

```cpp
class MyQueue {
  public:
    void push(int x) { inStack_.push(x); }

    int pop() {
        peek();
        int val = outStack_.top();
        outStack_.pop();
        return val;
    }

    int peek() {
        if (outStack_.empty()) {
            while (!inStack_.empty()) {
                outStack_.push(inStack_.top());
                inStack_.pop();
            }
        }
        return outStack_.top();
    }

    bool empty() { return inStack_.empty() && outStack_.empty(); }

  private:
    std::stack<int> inStack_;
    std::stack<int> outStack_;
};
```

- **复杂度**：`push` 时间 O(1)；`pop`/`peek` 摊还 O(1)；空间 O(n)。
- **易错点**：`empty` 必须同时判断两个栈，只看一个会在「已倒入 out_stack 后 in_stack 为空」时误判；`pop` 复用 `peek` 前要先确保 `out_stack` 已就绪（`peek` 会负责倒）；不要把「倒栈」放在 `push` 里，那会破坏顺序且增加成本。
- **相似题**：225. 用队列实现栈（本题的镜像，见下）；剑指 Offer 09. 用两个栈实现队列（同一题）；这类「受限容器互转」的思路在面试中常被用来考查对结构的理解。

### 225. 用队列实现栈（简单）

**题目**：请你仅使用两个队列实现一个后入先出栈，支持 `push(x)`（压入栈顶）、`pop()`（移除并返回栈顶元素）、`top()`（返回栈顶元素）、`empty()`（栈为空返回 `True`）。

**思路（用一个队列，每次把新元素转到队头）**：
其实一个队列就够了。`push(x)` 时先把 `x` 正常入队，再把队列中**除 `x` 以外的所有元素**
依次出队、重新入队。这样刚压入的 `x` 就被转到了队头，而队头正是栈顶。

为什么一轮旋转就能模拟栈：队列只能从尾进、从头出，天生先进先出。
但若每次新元素进来后，都把它前面已有的元素搬到它后面，队头就永远是最新加入的元素，
于是「后进」的元素先出，符合栈的语义。第一次搬 `n - 1` 个元素后队列整体转过一圈，
顺序恰好变成「新元素在最前，老的按入队顺序跟在后面」。

为什么让 `push` 多做、`pop` 不做：也可以反过来在 `pop` 时旋转。这里选择把成本放在 `push`，
好处是 `pop`/`top` 都只需读队头，代码短且对称；本质上是「用受限容器实现另一套语义」，
两种方案都正确。

**代码**（`src/stack/implement_stack_using_queues.py` / `.cpp`）：

```python
from collections import deque


class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return not self.q
```

```cpp
class MyStack {
  public:
    void push(int x) {
        q_.push(x);
        int n = q_.size();
        for (int i = 0; i < n - 1; ++i) {
            q_.push(q_.front());
            q_.pop();
        }
    }

    int pop() {
        int val = q_.front();
        q_.pop();
        return val;
    }

    int top() { return q_.front(); }

    bool empty() { return q_.empty(); }

  private:
    std::queue<int> q_;
};
```

- **复杂度**：`push` 时间 O(n)，`pop`/`top`/`empty` 时间 O(1)，空间 O(n)。
- **易错点**：旋转的轮数是「当前队列长度减一」，必须在入队**之后**重新取长度，否则会少转或多转；用 `deque` 而不是 list，`popleft` 才是 O(1)（list 的 `pop(0)` 是 O(n)，会让整体退化）；C++ 用 `std::queue`，只能访问 `front`，不能按下标取。
- **相似题**：232. 用栈实现队列（镜像题，见上）；两者对照着看，能体会到「栈和队列的区别只在取出端」；这类题在面试里用来确认你真正理解容器的进出规则。

---

## 规律总结

1. **栈解决的是「最近优先」的问题**。只要题目要求「回到最近一次未处理的状态」「后来者先配对/先撤销」，栈就是首选。20 的括号匹配、394 的解码、浏览器的后退，本质都是同一件事：最新的状态最容易被需要，而栈顶恰好就是它。

2. **匹配题的三步固定动作**：遇到「左半边」压栈；遇到「右半边」先判空、再比对栈顶、匹配则弹出否则失败；扫描结束检查栈为空。20 是这套动作最纯粹的模板，把这三步练熟，所有配对题都是换汤不换药。

3. **要 O(1) 查极值，就把极值缓存进辅助结构**。155 展示了通用手法：查询成本前移到更新时。再开一个栈保存「每一步的当前最小值」，主栈增删时同步维护。注意判据要用 `<=` 而非 `<`，否则重复极值会出错。这个「辅助结构跟着主结构同步走」的思想，在滑动窗口最大值（单调队列）里会再见。

4. **栈和队列只差一个「取出端」**。栈从同一端进出（LIFO），队列从两端进出（FIFO）。232 和 225 正是利用这个差别互相翻译：用两个栈「倒一次」得到队列顺序，用一个队列「转一圈」得到栈顺序。理解这类题的关键，是看清「哪一步把顺序反转了」。

5. **用受限容器模拟时，先想清楚成本放哪一端**。232 把倒栈放在弹出时（摊还 O(1)），225 把旋转放在压入时（O(n)）。两种做法都对，区别只是把复杂度记在谁头上。面试时先说明你的选择，再动手写。

6. **注意摊还复杂度与单次复杂度的区别**。232 的 `pop` 有时要搬很多元素、有时只用一次，但均摊下来是 O(1)；225 的 `push` 每次都是 O(n)。看到「有时慢、有时快」的代码，用摊还分析而不是最坏单次来评价它。

7. **`docs` 与 `src` 必须逐字一致**。题解里的代码与 `code/leetcode/src/stack/` 下的实现保持完全一致，以经过自测的 `src` 为准，文档只做粘贴，避免两边脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
