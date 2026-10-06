# 回溯与递归：决策树、排列与组合

回溯（backtracking）看起来变幻莫测，其实它只有一个内核：**把所有选择画成一棵树，
然后沿着「做选择 → 递归 → 撤销选择」的节奏，系统地走遍每个分支。**

想象你在一个岔路口，每条路代表一个选择。你挑一条走到底，如果走通了就记下结果；
走不通或者已经到头，就退回上一个岔路口换一条路——这个「退回来重选」的动作，
就是回溯里最关键的那一步。所以回溯的代码骨架永远长这样：

```text
def backtrack(路径, 选择列表):
    if 满足结束条件:
        收集结果
        return
    for 选择 in 选择列表:
        做选择
        backtrack(新路径, 新选择列表)
        撤销选择
```

**为什么必须在递归之后「撤销选择」**：`路径`和`选择列表`是同一条状态在递归栈上
共享的。进入下一层之前加上一个选择，从下一层回来之后就得把这个选择去掉，
否则它会污染兄弟分支。撤销这一步，是回溯区别于普通 DFS 的标记。

本篇用三道最经典的题，把回溯分成两族来对照：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：组合与子集（无顺序，用 `start` 只往后选） | 78. 子集 | 中等 |
| 模式一：组合与子集（无顺序，用 `start` 只往后选） | 77. 组合 | 中等 |
| 模式二：全排列（有顺序，用 `used` 标记） | 46. 全排列 | 中等 |

---

## 模式一：组合与子集——用 `start` 保证「只往后选」

**适用信号**：题目问「有哪些子集 / 选出 k 个数 / 元素组合」，而且
`[1, 2]` 和 `[2, 1]` 被当成**同一个结果**。关键词常带「组合 / 子集 / 选取」。

**核心动作**：递归函数带一个参数 `start`，表示「这一层只能从下标 `start` 往后挑」。
选了 `nums[i]` 之后，下一层从 `i + 1` 开始——既不会回头选到已经用过的元素，
也不会用不同的顺序重复生成同一个组合。

**为什么 `start` 能去重**：规定元素必须按原数组下标的递增顺序被选取，
任何一组「元素集合」都只有一种符合该顺序的排法，于是每种组合只会被生成一次。
组合题去重的第一招，往往不是哈希表，而是这个「强制有序」的约定。

### 78. 子集（中等）

**题目**：给你一个元素互不相同的整数数组 `nums`，返回它所有可能的子集（幂集）。
解集不能包含重复的子集，可以按任意顺序返回。

**思路**：
把「造一个子集」想成一棵决策树：从下标 0 出发，每一步决定「要不要选当前位置的数」。
这棵树上**每一个节点**都对应一个合法子集——根是空集，每往下选一个数就多一个元素。
所以和常见的回溯不同，这里**不只在叶子收集结果，而是每进入一次递归就收集一次**。

外层循环用 `start` 控制可选范围：选完 `nums[i]` 后只能从 `i + 1` 继续，
从而保证每个子集只被生成一次。

**为什么每个节点都要收**：子集不像组合有固定长度，任何「选了一部分的中间状态」
本身就是答案。根节点（什么都不选）也是答案之一。

**代码**（完整可运行版见 `src/backtracking/subsets.py` / `.cpp`）：

```python
def subsets(nums):
    res = []
    path = []

    def backtrack(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res
```

```cpp
void backtrack(const std::vector<int> &nums, int start, std::vector<int> &path,
               std::vector<std::vector<int>> &res) {
    res.push_back(path);
    for (int i = start; i < static_cast<int>(nums.size()); ++i) {
        path.push_back(nums[i]);
        backtrack(nums, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> subsets(const std::vector<int> &nums) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(nums, 0, path, res);
    return res;
}
```

- **复杂度**：时间 O(n·2^n)，空间 O(n)（递归深度，不含答案本身）。
- **易错点**：把 `res.append(path[:])` 写成 `res.append(path)`——后者存的是引用，
  后续 `path` 变化会连带改掉答案，而 `path[:]` 复制了一份快照；
  忘记在循环后 `path.pop()`，路径会越积越长；
  下一层传 `i + 1` 而不是 `i`，传 `i` 会导致同一个元素被反复选取。
- **相似题**：77. 组合（固定长度版本）、90. 子集 II（数组含重复元素，见后续
  回溯专题的「排序 + 同层去重」）。

### 77. 组合（中等）

**题目**：给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。
可以按任意顺序返回答案。

**思路**：
这道题和子集共用同一棵决策树，唯一的不同是**收集结果的时机**：子集在每个节点都收，
组合只在 `path` 长度凑够 `k` 时才收，那时候就返回，不必再往下选。

同样用 `start` 保证只往后选，避免 `[1, 4]` 和 `[4, 1]` 重复出现。
在此基础上还能剪一刀：**如果剩下的数全选上都不够 k 个，就整段跳过**。
当前已选 `len(path)` 个，还差 `k - len(path)` 个；从 `i` 到 `n` 一共
`n - i + 1` 个数，要让它至少等于还差的数量，解得循环上界
`i ≤ n - (k - len(path)) + 1`。

**为什么剪枝是对的**：被剪掉的那些分支，无论如何都凑不齐 `k` 个数，
走它们只是白费递归。剪枝不改变答案，只改变搜索的规模，是回溯效率的主要来源。

**代码**（`src/backtracking/combinations.py` / `.cpp`）：

```python
def combine(n, k):
    res = []
    path = []

    def backtrack(start):
        if len(path) == k:
            res.append(path[:])
            return
        need = k - len(path)
        for i in range(start, n - need + 2):
            path.append(i)
            backtrack(i + 1)
            path.pop()

    backtrack(1)
    return res
```

```cpp
void backtrack(int n, int k, int start, std::vector<int> &path,
               std::vector<std::vector<int>> &res) {
    if (static_cast<int>(path.size()) == k) {
        res.push_back(path);
        return;
    }
    int need = k - static_cast<int>(path.size());
    for (int i = start; i <= n - need + 1; ++i) {
        path.push_back(i);
        backtrack(n, k, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combine(int n, int k) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(n, k, 1, path, res);
    return res;
}
```

- **复杂度**：时间 O(C(n,k)·k)，空间 O(k)（递归深度，不含答案本身）。
- **易错点**：范围是 `[1, n]`，起始值从 1 开始，别写成 0；
  剪枝上界写成 `n - need + 2` 而不是 `n - need + 1`——因为 Python 的
  `range` 右端开区间，加 1 才是闭区间；C++ 的 `for` 是 `<=`，所以写 `+ 1`，
  两者等价但形式不同，照抄容易混；
  收集后要 `return`，否则会继续往 `path` 里塞超过 `k` 个元素。
- **相似题**：78. 子集（不限长度）、39 / 40. 组合总和（允许重复选取 / 含重复元素，
  见后续回溯专题）。这三个一起练，就能体会「同一棵树，只是收集时机和可选范围不同」。

---

## 模式二：全排列——用 `used` 标记「已经用过的元素」

**适用信号**：题目问「所有排列 / 所有顺序」，`[1, 2]` 和 `[2, 1]` 是**两个不同结果**。
关键词常带「排列 / 顺序 / 全排列」。

**核心动作**：排列看重顺序，所以每一层都从下标 0 重新挑，不能再用 `start` 限制范围。
改用布尔数组 `used`（或哈希集合）记录哪些元素已经被当前路径用过，跳过用过的。
当 `path` 的长度等于数组长度时，说明每个元素都用上了，得到一个完整排列。

**为什么不能沿用 `start`**：`start` 的本质是「强制元素按原顺序出现」，那正好会
把 `[2, 1]` 这样的逆序排列挡在门外。排列要的恰恰是「顺序自由」，所以必须换一种
「只排除已用过的、不限制顺序」的机制，这就是 `used`。

### 46. 全排列（中等）

**题目**：给定一个不含重复数字的数组 `nums`，返回其所有可能的全排列。
可以按任意顺序返回答案。

**思路**：
决策树上，每一层决定「下一个位置放哪个还没用过的数」。用 `used[i]` 表示 `nums[i]`
是否已经在当前路径里：遍历所有下标，跳过 `used[i]` 的，剩下的逐个尝试。
选一个就 `used[i] = True` 并压入 `path`，递归返回后**一定要把两者都还原**。

路径长度到 `n` 时，所有数都被用完，此时 `path` 就是一个完整排列，复制进答案。

**为什么必须撤销 `used`**：`path` 与 `used` 是同一条状态的两面，选的时候一起变，
回溯的时候也必须一起还原。只 `pop` 而不把 `used[i]` 复位，会让这个数在后续
兄弟分支里「再也用不了」，结果漏解。

**代码**（`src/backtracking/permutations.py` / `.cpp`）：

```python
def permute(nums):
    res = []
    path = []
    used = [False] * len(nums)

    def backtrack():
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack()
            path.pop()
            used[i] = False

    backtrack()
    return res
```

```cpp
void backtrack(const std::vector<int> &nums, std::vector<int> &path,
               std::vector<bool> &used, std::vector<std::vector<int>> &res) {
    if (path.size() == nums.size()) {
        res.push_back(path);
        return;
    }
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        if (used[i]) continue;
        used[i] = true;
        path.push_back(nums[i]);
        backtrack(nums, path, used, res);
        path.pop_back();
        used[i] = false;
    }
}

std::vector<std::vector<int>> permute(const std::vector<int> &nums) {
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    std::vector<bool> used(nums.size(), false);
    backtrack(nums, path, used, res);
    return res;
}
```

- **复杂度**：时间 O(n·n!)，空间 O(n)（递归深度 + `used`，不含答案本身）。
- **易错点**：循环从 0 开始（不是 `start`），这是与组合/子集最大的区别；
  `used[i]` 的复位不能漏，否则兄弟分支少算；
  终止条件是 `len(path) == len(nums)`，别写成 `== used.count(True)` 之类的啰嗦写法；
  `path[:]` 复制快照，别存引用。
- **相似题**：47. 全排列 II（数组含重复元素，排序后用「同层去重」跳过重复分支）、
  78. 子集、77. 组合（无顺序族，与本族对照记忆）。

---

## 规律总结

1. **回溯的统一骨架：做选择 → 递归 → 撤销选择**。把问题想成一棵决策树，
   `path` 记录当前路径，循环枚举可选分支，进入下一层前改状态、回来后还原状态。
   「撤销选择」是回溯的灵魂，漏了它兄弟分支就会被污染。

2. **先判断「顺序是否有意义」，再决定用 `start` 还是 `used`**。
   - 顺序无关（组合、子集，`[1,2]` 等于 `[2,1]`）→ 用 **`start` 只往后选**，
     强制元素按下标递增出现，天然去重；
   - 顺序有关（排列，`[1,2]` 不同于 `[2,1]`）→ 用 **`used` 标记已用元素**，
     每层从 0 重新挑，只排除用过的。

3. **收集结果的时机取决于题目**：子集在每个节点都收（含根的空集）；
   组合、排列只在路径达到目标长度时收。想清楚「哪些状态算答案」，
   收集语句的位置就确定了。

4. **组合/排列的去重与剪枝**：重复元素的去重靠「排序 + 同一层跳过相同值」
   （见后续 90 / 47 题）；剪枝靠「剩余元素不够凑齐目标就直接跳过」
   （见 77 的循环上界）。剪枝不改变答案，只砍掉注定失败的子树。

5. **结果的复制不能省**：`path` 是会被反复修改的共享变量，收集时必须
   `res.append(path[:])`（Python）或 `res.push_back(path)`（C++，
   `vector` 是值拷贝）。存引用会让所有答案最终都变成最后一次的状态。

6. **复杂度通常是指数级**：排列 O(n·n!)，子集 O(n·2^n)，组合 O(C(n,k)·k)。
   递归深度是 O(n)，若答案集本身很大，输出规模就是复杂度下界。
   回溯的优化方向只有一类——**剪枝**，即尽早识别并放弃注定失败的分支。

7. **和 DFS 是一体两面**：回溯就是「在决策树上做 DFS + 状态撤销」。
   第 10 篇讲的是在图的节点上走，这一篇讲的是在「选择空间」里走，
   骨架（进入 → 递归邻接 → 退出）完全相通。
