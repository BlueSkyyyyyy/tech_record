# 栈与单调栈

栈是另一种「线性表」，但规矩比数组和链表都严格：**只能在一端进出，后进先出（LIFO）**。
正因为它这么「死板」，反而特别适合处理一类特定问题——**需要“回到最近一次未处理的状态”**。
生活里最直观的例子是浏览器的后退、编辑器的撤销：最新的那一步总是最先被撤销。

栈的价值不在存储，而在「顺序」。当问题的处理顺序天然是「后来的先处理」时，栈往往能让代码变得极短：
括号匹配、表达式求值、单调栈找下一个更大元素，都源于这一点。本篇先用四道基础题把栈
「怎么存、怎么用、怎么和别的结构互转」讲清楚，再进入本篇的重头戏——**单调栈**：
它是一种「让栈里的元素始终保持单调」的技巧，专门用来批量求解「每个元素左边/右边第一个更大（更小）的元素」，
每日温度、下一个更大元素、柱状图最大矩形，本质都是同一套模板。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 匹配与消消看 | 20. 有效的括号 | 简单 |
| 辅助栈记录极值 | 155. 最小栈 | 简单 |
| 受限容器互相模拟 | 232. 用栈实现队列 | 简单 |
| 受限容器互相模拟 | 225. 用队列实现栈 | 简单 |
| 后缀表达式 | 150. 逆波兰表达式求值 | 中等 |
| 单调栈（下一个更大） | 739. 每日温度 | 中等 |
| 单调栈（下一个更大） | 496. 下一个更大元素 I | 简单 |
| 单调栈（下一个更大） | 503. 下一个更大元素 II | 中等 |
| 单调栈（最大矩形） | 84. 柱状图中最大的矩形 | 困难 |
| 单调栈（最大矩形） | 42. 接雨水（单调栈解） | 困难 |
| 嵌套结构解码 | 394. 字符串解码 | 中等 |

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

## 模式四：后缀表达式求值

**适用信号**：题目直接给出**后缀（逆波兰）表达式**，或让你从表达式中缀转后缀。
关键词是「逆波兰」「后缀」「运算符在两个操作数之后」。
核心动作：**用一个栈存操作数**，遇到数字就压、遇到运算符就弹出两个数算完再压回。

### 150. 逆波兰表达式求值（中等）

**题目**：给定一个按逆波兰表示法（后缀表达式）排列的字符串数组 `tokens`，求该表达式的值。合法的运算符只有 `+`、`-`、`*`、`/`，除法向零截断。

**思路（一个栈存操作数）**：
从左到右读 token。遇到数字就压栈；遇到运算符就弹出栈顶两个数，**先弹出的是右操作数 `b`、后弹出的是左操作数 `a`**，算完 `a op b` 再压回栈。扫完后栈里只剩一个数，就是答案。

为什么用栈：后缀表达式把运算顺序「藏」在了 token 的先后里，每次运算的两个操作数，正是**最近两个还没被消耗的数**——最近优先，就是栈顶。先遇到的数先入栈、沉得深，后遇到的数在栈顶，所以做运算时先弹出的是右操作数。这个「先弹 b、再弹 a」的顺序是本题最容易写反的地方（减法和除法不满足交换律）。

为什么除法要特别处理：LeetCode 规定除法**向零截断**（如 `-7 / 2 = -3`），而 Python 的 `//` 是向下取整（`-7 // 2 = -4`），所以这里用 `int(a / b)`；C++ 的整数除法本来就是向零截断，直接 `a / b` 即可。

**代码**（`src/stack/eval_rpn.py` / `.cpp`）：

```python
def eval_rpn(tokens):
    stack = []
    ops = {"+", "-", "*", "/"}
    for tok in tokens:
        if tok in ops:
            b = stack.pop()
            a = stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(tok))
    return stack[-1]
```

```cpp
int evalRPN(const std::vector<std::string> &tokens) {
    std::stack<int> st;
    for (const std::string &tok : tokens) {
        if (tok == "+" || tok == "-" || tok == "*" || tok == "/") {
            int b = st.top();
            st.pop();
            int a = st.top();
            st.pop();
            if (tok == "+") st.push(a + b);
            else if (tok == "-") st.push(a - b);
            else if (tok == "*") st.push(a * b);
            else st.push(a / b);  // C++ 整数除法天然向零截断
        } else {
            st.push(std::stoi(tok));
        }
    }
    return st.top();
}
```

- **复杂度**：时间 O(n)（每个 token 处理一次），空间 O(n)（最坏全是数字）。
- **易错点**：弹出顺序是「先 b 后 a」，减法和除法写反会得到相反的结果；Python 别用 `//`，负数的截断方向不对；token 是字符串，取数字前要转成整数。
- **相似题**：224. 基本计算器（中缀表达式，需要额外处理括号与正负号）；227. 基本计算器 II（中缀，用栈处理优先级）；150 是表达式求值里最简单的一环——先学后缀，再回头处理中缀。

---

## 模式五：单调栈找「下一个更大元素」

**适用信号**：题目要你**对每个元素，找它左边/右边第一个比它大（或小）的元素**，或批量求「等多少天」「隔多远」。关键词是「下一个更大」「最近一个更高」。
核心动作：维护一个**单调栈**——栈里保存「还没找到答案」的元素，它们的值是单调的。一旦新来的元素能给出答案，就连续弹出结算。

理解单调栈最关键的一点：**栈里放的是「悬而未决」的元素**，而不是苦苦等待。新元素若能满足栈顶，往往也能满足栈里更靠内的若干元素（因为它们更小、更靠左），于是一次 `while` 就把一批答案同时算完，这就是它比暴力 O(n²) 快的原因。

### 739. 每日温度（中等）

**题目**：给定整数数组 `temperatures`，返回数组 `answer`，其中 `answer[i]` 表示第 `i` 天之后要等多少天才会出现更高的温度；若之后都不会更高，填 `0`。

**思路（单调递减栈，栈里存下标）**：
维护一个「温度单调递减」的下标栈。每遇到当天温度 `t`，只要栈顶那天的温度比 `t` 低，就说明栈顶那天等到了更高的温度：答案是 `i - 栈顶下标`，弹掉它。重复直到栈空或栈顶温度不低于 `t`，再把当天压栈。

为什么栈里存下标不存温度：答案要的是「相差多少天」，必须知道位置；存下标即可同时取温度。为什么一次弹出能结算一批：栈内温度从底到顶递减，新来的 `t` 比栈顶高时，栈里比 `t` 低的那些天也都等到了 `t`，而且它们更靠左，所以可以连续结算。每个下标最多进栈出栈一次，整体 O(n)。

**代码**（`src/stack/daily_temperatures.py` / `.cpp`）：

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    answer = [0] * n
    stack = []
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer
```

```cpp
std::vector<int> dailyTemperatures(const std::vector<int> &temperatures) {
    int n = temperatures.size();
    std::vector<int> answer(n, 0);
    std::stack<int> st;
    for (int i = 0; i < n; ++i) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int j = st.top();
            st.pop();
            answer[j] = i - j;
        }
        st.push(i);
    }
    return answer;
}
```

- **复杂度**：时间 O(n)（每个下标进出栈一次），空间 O(n)。
- **易错点**：比较的是「栈顶那天的温度」而不是栈顶下标本身，别写成 `st.top() < t`；答案初始化为 0，没有更高温的日子自然保持 0；判据是严格 `<`（相等不算更高）。
- **相似题**：496. 下一个更大元素 I（同一模板，只是把答案存进哈希表，见下）；503. 下一个更大元素 II（循环数组版，见下）；注意本题与「接雨水」的区别——这里是找严格更大的第一个元素，接雨水是找两侧更高的围栏。

### 496. 下一个更大元素 I（简单）

**题目**：`nums1` 是 `nums2` 的子集。对 `nums1` 中每个元素 `x`，找出它在 `nums2` 中对应位置右侧第一个比它大的元素；不存在则输出 `-1`。

**思路（先在 `nums2` 上跑单调栈，把「每个值 → 它的下一个更大值」存进哈希表）**：
遍历 `nums2`，维护一个单调递减栈。遇到比栈顶大的 `x`，就不断弹出栈顶 `v`，记下 `next_greater[v] = x`；最后把 `x` 压栈。遍历完后，哈希表里就装好了 `nums2` 中所有「有下一个更大值」的元素的答案。再按 `nums1` 的顺序查表，查不到就是 `-1`。

为什么可以先只处理 `nums2`：`nums1` 只是 `nums2` 的一个查询子集。与其对 `nums1` 每个元素都去 `nums2` 里找位置再向后扫（慢），不如一次把 `nums2` 的答案全算好，之后每次查询都是 O(1)——典型的「预处理 + 哈希查询」空间换时间。这里栈里存「值」而不是下标，是因为输出的是值本身，且 `nums2` 无重复元素，不必靠下标区分。

**代码**（`src/stack/next_greater_element_i.py` / `.cpp`）：

```python
def next_greater_element(nums1, nums2):
    next_greater = {}
    stack = []
    for x in nums2:
        while stack and stack[-1] < x:
            next_greater[stack.pop()] = x
        stack.append(x)
    return [next_greater.get(x, -1) for x in nums1]
```

```cpp
std::vector<int> nextGreaterElement(const std::vector<int> &nums1,
                                    const std::vector<int> &nums2) {
    std::unordered_map<int, int> nextGreater;
    std::stack<int> st;
    for (int x : nums2) {
        while (!st.empty() && st.top() < x) {
            nextGreater[st.top()] = x;
            st.pop();
        }
        st.push(x);
    }
    std::vector<int> result;
    for (int x : nums1) {
        auto it = nextGreater.find(x);
        result.push_back(it == nextGreater.end() ? -1 : it->second);
    }
    return result;
}
```

- **复杂度**：时间 O(n + m)（n、m 分别是 `nums2`、`nums1` 长度），空间 O(n)。
- **易错点**：查表用 `get(x, -1)` / `find`，别直接下标访问，否则缺失的 key 会报错；只有 `nums2` 里确实有更大值的元素才会进哈希表，其余默认 -1；`nums1` 是子集这一条件保证了查询一定能在 `nums2` 中找到对应元素。
- **相似题**：739. 每日温度（把答案改成「距离」）；503. 下一个更大元素 II（循环版，见下）；496 与 503 只差「数组是否首尾相接」，模板完全一致。

### 503. 下一个更大元素 II（中等）

**题目**：给定一个**循环数组** `nums`（最后一个元素的下一个元素是数组的第一个元素），返回 `nums` 中每个元素的下一个更大元素；不存在则输出 `-1`。

**思路（单调栈 + 把数组「逻辑上接长一倍」）**：
循环数组的麻烦在于，一个元素的更大值可能出现在它左边。解决办法是**逻辑上把数组遍历两遍**：用下标 `i` 从 0 走到 `2n-1`，真实元素取 `nums[i % n]`。第一遍负责建立栈、结算第一遍里能确定的答案；第二遍让「绕回开头」的元素有机会去结算那些一直没等到更大值的元素。

关键细节：只有 `i < n` 时才把下标压栈。第二遍是「补算」用的，不能再往里塞重复下标，否则同一位置会被处理两次，还会让栈无限增长。为什么遍历两圈就够：任意元素的下一个更大值，要么在它右侧（第一圈就能找到），要么在它左侧、需要绕一圈（第二圈覆盖）。两圈后仍没结算的，就是真的没有更大值，保留初始的 `-1`。

**代码**（`src/stack/next_greater_element_ii.py` / `.cpp`）：

```python
def next_greater_elements(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        x = nums[i % n]
        while stack and nums[stack[-1]] < x:
            result[stack.pop()] = x
        if i < n:
            stack.append(i)
    return result
```

```cpp
std::vector<int> nextGreaterElements(const std::vector<int> &nums) {
    int n = nums.size();
    std::vector<int> result(n, -1);
    std::stack<int> st;
    for (int i = 0; i < 2 * n; ++i) {
        int x = nums[i % n];
        while (!st.empty() && nums[st.top()] < x) {
            result[st.top()] = x;
            st.pop();
        }
        if (i < n) st.push(i);
    }
    return result;
}
```

- **复杂度**：时间 O(n)（每个下标最多进出栈一次），空间 O(n)。
- **易错点**：忘了 `i < n` 的压栈限制，会导致第二圈重复压入、结果错乱；`i % n` 用来取真实元素，别写成 `nums[i]` 越界；result 初始化为 -1，别用 0（元素可能为负）。
- **相似题**：496. 下一个更大元素 I（非循环版，见上）；739. 每日温度（同一模板的「距离」变体）；把循环展开成「两倍长度」是处理环形数组的通用手法，环形链表的 142 也是类似思路（绕圈相会）。

---

## 模式六：单调栈求「最大矩形 / 凹槽蓄水」

**适用信号**：题目让你在柱状图/直方图里找**最大矩形面积**，或在高度图里算**能接多少水**。关键词是「柱状图」「矩形」「接雨水」「凹槽」。
核心动作：**枚举「以某根柱子为高（或为槽底）」**，用单调栈找它左右两侧第一个更矮（或更高）的柱子作为边界，从而 O(1) 结算这块面积的宽度。

### 84. 柱状图中最大的矩形（困难）

**题目**：给定 `n` 个非负整数表示柱状图中各柱子的高度（宽度均为 1），求该柱状图内能勾勒出的最大矩形面积。

**思路（单调递增栈，栈里存下标）**：
枚举「以某根柱子为矩形的高」。此时矩形的左右边界，是它左右两边**第一个比它矮的柱子**——因为一旦碰到更矮的柱子，高度就维持不住了。于是问题变成：对每根柱子，找它左边和右边第一个更矮的位置，这正是单调栈擅长的。

从左到右扫描，维护一个高度单调递增的栈。遇到当前高度 `h` 比栈顶矮时，栈顶那根柱子「右边第一个更矮者」就是当前 `i`，它「左边第一个更矮者」就是栈里它下面那根，两者之间的宽度 `i - stack[-1] - 1`（栈空则为 `i`）就是它能撑起的最大宽度，乘上它的高度就是这块面积，弹出结算。技巧：在数组末尾补一个高度 0 的哨兵，这样扫描结束时栈里所有柱子都会遇到「更矮的 0」而被结算，不必再写一段收尾代码。

**代码**（`src/stack/largest_rectangle_in_histogram.py` / `.cpp`）：

```python
def largest_rectangle_area(heights):
    heights = list(heights) + [0]
    stack = []
    best = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            bar = stack.pop()
            height = heights[bar]
            left = stack[-1] if stack else -1
            width = i - left - 1
            best = max(best, height * width)
        stack.append(i)
    return best
```

```cpp
int largestRectangleArea(std::vector<int> heights) {
    heights.push_back(0);  // 末尾哨兵，保证收尾时全部结算
    std::stack<int> st;
    int best = 0;
    for (int i = 0; i < (int)heights.size(); ++i) {
        while (!st.empty() && heights[st.top()] > heights[i]) {
            int bar = st.top();
            st.pop();
            int height = heights[bar];
            int left = st.empty() ? -1 : st.top();
            int width = i - left - 1;
            best = std::max(best, height * width);
        }
        st.push(i);
    }
    return best;
}
```

- **复杂度**：时间 O(n)（每个下标进出栈一次），空间 O(n)（栈 + 一份拷贝）。
- **易错点**：宽度公式是 `i - left - 1`，`left` 取「栈内下一个下标」，栈空时取 -1；末尾哨兵是收尾的关键，少了它会漏算仍在栈里的柱子；Python 里先把 `heights` 拷贝再 `append`，别原地修改调用者的数组。
- **相似题**：42. 接雨水（同样的凹槽思路，见下）；85. 最大矩形（把二维矩阵按行压成一维直方图，再套本题）；11. 盛最多水的容器（对撞双指针版，与本题的单调栈是两种思路，见 `array`）。

### 42. 接雨水（困难，单调栈解，与 `array` 交叉）

**题目**：给定 `n` 个非负整数表示每个宽度为 1 的柱子的高度图，计算下雨之后能接多少雨水。

**思路（单调递减栈，横向累加水）**：
维护一个高度递减的下标栈。当遇到比栈顶更高的柱子 `i` 时，栈顶那根就是「凹槽底」，它与当前柱子之间形成了一个能蓄水的坑。弹出槽底 `bottom` 后，若栈还不空，栈顶 `left` 就是坑的左壁，右壁是当前 `i`，宽度是 `i - left - 1`；水位由较矮的那面墙决定，可蓄高度是 `min(height[left], height[i]) - height[bottom]`，两者相乘即这一层的积水量，累加即可。

为什么是「横向按层累加」：同一个凹槽可能被更高的柱子分成多层，每弹出一个槽底结算一层，正好把坑填满。与双指针解的区别：双指针是「纵向」逐个位置算它上方能存多高，O(1) 空间；单调栈是「横向」按层算，需要 O(n) 栈空间。两种都对，`array` 篇用的是双指针版（代码见 `src/array/trapping_rain_water.py`）。

**代码**（单调栈解见 `src/array/trapping_rain_water.py` 的 `trap_stack`）：

```python
def trap_stack(height):
    stack = []
    water = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            bounded = min(height[stack[-1]], h) - height[bottom]
            water += width * bounded
        stack.append(i)
    return water
```

- **复杂度**：时间 O(n)，空间 O(n)（栈）。若改用双指针则为 O(n)/O(1)。
- **易错点**：弹出槽底后必须判断栈是否为空——空了说明左边没有墙，形不成坑，直接退出；水位用 `min(左壁, 右壁) - 槽底`，可能为 0（相邻等高时这一层不蓄水），不必特判。
- **相似题**：84. 柱状图中最大的矩形（单调栈求面积，见上）；11. 盛最多水的容器（`array`，对撞双指针）；407. 接雨水 II（二维版，用优先队列从边界向内收缩）。

---

## 模式七：栈处理嵌套结构

**适用信号**：输入是**层层嵌套**的括号结构（如 `3[a2[c]]`），需要**从里到外**依次解析。关键词是「嵌套」「括号」「解码」「展开」。
核心动作：遇到「开始符」时把**外层上下文压栈暂存**，遇到「结束符」时取回并合并内层结果——栈把「先遇到外层、后处理内层」的顺序倒成需要的顺序。

### 394. 字符串解码（中等）

**题目**：给定一个编码字符串，形如 `k[encoded_string]`，表示方括号里的内容重复 `k` 次，可以嵌套（如 `3[a2[c]]` 先解成 `acc`，再整体重复 3 次得 `accaccacc`）。返回解码后的字符串。

**思路（两个栈，一个存「外层已拼好的串」，一个存重复次数）**：
从左到右扫描：遇到数字就累积当前数字 `num`（可能多位，用 `num = num * 10 + digit`）；遇到 `[` 就把「当前已拼好的串 `cur`」和「当前重复次数 `num`」一起压栈，然后清空 `cur`、`num`，开始处理括号里的内容；遇到 `]` 就弹出外层串 `prev` 和次数 `repeat`，把括号内的 `cur` 重复 `repeat` 次，再拼回 `prev` 作为新的 `cur`；遇到字母直接接到 `cur` 后面。

为什么用栈：解码是从里到外的——必须先解出最内层方括号，才能去重复外层。但扫描是从左到右、先遇到外层 `[`。栈正好把这个顺序倒过来：`[` 时把外层上下文暂存，`]` 时取回，天然匹配「后进先出」的嵌套结构。数字单独用累加而不是直接转整数，是因为 `k` 可能是多位数，必须等遇到 `[` 才知道数字读完了。

**代码**（`src/stack/decode_string.py` / `.cpp`）：

```python
def decode_string(s):
    stack = []
    cur = ""
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append((cur, num))
            cur = ""
            num = 0
        elif ch == "]":
            prev, repeat = stack.pop()
            cur = prev + cur * repeat
        else:
            cur += ch
    return cur
```

```cpp
std::string decodeString(const std::string &s) {
    std::stack<std::pair<std::string, int>> st;
    std::string cur;
    int num = 0;
    for (char ch : s) {
        if (std::isdigit((unsigned char)ch)) {
            num = num * 10 + (ch - '0');
        } else if (ch == '[') {
            st.push({cur, num});
            cur.clear();
            num = 0;
        } else if (ch == ']') {
            auto [prev, repeat] = st.top();
            st.pop();
            std::string expanded;
            for (int i = 0; i < repeat; ++i) expanded += cur;
            cur = prev + expanded;
        } else {
            cur += ch;
        }
    }
    return cur;
}
```

- **复杂度**：时间 O(解码后串长)（每个字符被处理的次数与其最终出现次数同一量级），空间 O(解码后串长)（中间结果）。
- **易错点**：数字可能多位，必须用 `num = num * 10 + digit` 累积，遇到 `[` 时才归零；`[` 入栈的是「外层串 + 当前次数」，顺序别搞反；`]` 时是 `prev + cur * repeat`（把内层重复后拼到外层后面），不是相乘；C++ 里 `std::isdigit` 的参数要转 `unsigned char`，避免负值未定义行为。
- **相似题**：20. 有效的括号（同一个「嵌套 + 栈」思想的最简版，见 `07-stack` 模式一）；726. 原子的数量（也是嵌套括号解析，额外要合并同类项并排序）；394 是嵌套解析里最好上手的模板，练熟后 726 只是多了一个「计数并排序」的收尾。

---

## 规律总结

1. **栈解决的是「最近优先」的问题**。只要题目要求「回到最近一次未处理的状态」「后来者先配对/先撤销」，栈就是首选。20 的括号匹配、394 的解码、浏览器的后退，本质都是同一件事：最新的状态最容易被需要，而栈顶恰好就是它。

2. **匹配题的三步固定动作**：遇到「左半边」压栈；遇到「右半边」先判空、再比对栈顶、匹配则弹出否则失败；扫描结束检查栈为空。20 是这套动作最纯粹的模板，把这三步练熟，所有配对题都是换汤不换药。

3. **要 O(1) 查极值，就把极值缓存进辅助结构**。155 展示了通用手法：查询成本前移到更新时。再开一个栈保存「每一步的当前最小值」，主栈增删时同步维护。注意判据要用 `<=` 而非 `<`，否则重复极值会出错。这个「辅助结构跟着主结构同步走」的思想，在滑动窗口最大值（单调队列）里会再见。

4. **栈和队列只差一个「取出端」**。栈从同一端进出（LIFO），队列从两端进出（FIFO）。232 和 225 正是利用这个差别互相翻译：用两个栈「倒一次」得到队列顺序，用一个队列「转一圈」得到栈顺序。理解这类题的关键，是看清「哪一步把顺序反转了」。

5. **用受限容器模拟时，先想清楚成本放哪一端**。232 把倒栈放在弹出时（摊还 O(1)），225 把旋转放在压入时（O(n)）。两种做法都对，区别只是把复杂度记在谁头上。面试时先说明你的选择，再动手写。

6. **注意摊还复杂度与单次复杂度的区别**。232 的 `pop` 有时要搬很多元素、有时只用一次，但均摊下来是 O(1)；225 的 `push` 每次都是 O(n)。看到「有时慢、有时快」的代码，用摊还分析而不是最坏单次来评价它。

7. **单调栈的通用模板**：遍历数组，对每个元素 `x`，`while (栈非空 && 栈顶不满足单调性) 弹出并结算; 压入 x`。它专门批量求「每个元素左/右第一个更大（更小）的元素」。判断方向的口诀是：**栈里保存的是「还没找到答案、正在等待」的元素**，所以它们的值是单调的；新元素一来就能一次性结算栈顶的一批。739、496、503 是同一套模板的三种包装（距离 / 查表 / 循环数组），84、42 则是把「找更矮的边界」用于面积计算。

8. **单调栈为什么要存下标而不是值**：一旦答案依赖「位置」（差多少天、宽度是多少），就必须存下标；只有当输出就是值本身、且数组无重复元素时（如 496）才可以省掉下标。769 这类不涉及距离的题也常用值，但要先确认无重复。

9. **环形数组用「遍历两遍 + 取模」破解**。503 把长度 `n` 的循环数组扫 `2n` 次，`nums[i % n]` 取真实元素，且只在 `i < n` 时压栈。这样「绕回左边」的更大值在第二圈被找到，而不必真的复制数组。任何「首尾相接」的题（环形链表、环形子数组）都可以先想这个「接长一倍」的思路。

10. **凹槽类问题的两种视角**：双指针是「纵向逐格」算每个位置上方能存多少水（O(1) 空间），单调栈是「横向按层」弹出槽底、一层层累加（O(n) 空间）。42 两种实现都给了，对照着看能明白：同一道题，选择不同的枚举对象，就得到不同复杂度的解法。

11. **`docs` 与 `src` 必须逐字一致**。题解里的代码与 `code/leetcode/src/stack/` 下的实现保持完全一致，以经过自测的 `src` 为准，文档只做粘贴，避免两边脱节。C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被当成宏参数分隔符，先把期望值存进变量再比较。
