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

本篇用十二道经典题，把回溯由浅入深地铺开：先立起「组合 / 子集 / 排列」两族基本盘，
再依次加上「有重复元素怎么去重」「目标和组合」「网格与字符串切割」「带约束的构造」
四类新场景。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：组合与子集（无顺序，用 `start` 只往后选） | 78. 子集 | 中等 |
| 模式一：组合与子集（无顺序，用 `start` 只往后选） | 77. 组合 | 中等 |
| 模式二：全排列（有顺序，用 `used` 标记） | 46. 全排列 | 中等 |
| 模式三：含重复元素的去重（排序 + 同层跳过） | 90. 子集 II | 中等 |
| 模式三：含重复元素的去重（排序 + 同层跳过） | 47. 全排列 II | 中等 |
| 模式四：目标和的组合（可重复选取 / 每数一次） | 39. 组合总和 | 中等 |
| 模式四：目标和的组合（可重复选取 / 每数一次） | 40. 组合总和 II | 中等 |
| 模式五：把决策树铺到网格 / 字符串上 | 79. 单词搜索 | 中等 |
| 模式五：把决策树铺到网格 / 字符串上 | 131. 分割回文串 | 中等 |
| 模式六：带约束的构造 | 17. 电话号码的字母组合 | 中等 |
| 模式六：带约束的构造 | 22. 括号生成 | 中等 |
| 模式六：带约束的构造 | 51. N 皇后 | 困难 |

前四个模式是「排列组合」的原地深化，后两个模式则把同一套骨架搬到网格、字符串和
约束构造这些不同舞台。读的时候不妨始终问自己三个问题：**决策树的每个节点代表什么？
收集答案在什么时候？哪些分支可以提前砍掉？**

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

## 模式三：含重复元素的去重——排序 + 同层跳过

**适用信号**：题目里的数组「可能包含重复元素」，而答案要求「不重复」的组合 /
子集 / 排列。这是 78 / 77 / 46 的进阶版，难点从「怎么枚举」变成了「怎么不重复」。

**核心动作**：先 `sort` 让相同的数挨在一起，再在每层循环里加一句判据。子集与组合
用「`i > start` 且 `nums[i] == nums[i-1]` 就跳过」；排列的判据多一个条件，见 47。

**为什么排序能帮上忙**：重复只可能来自「同一个数值被选两次、但走的是不同下标」。
排序把相同的值聚到一起后，只要规定「同一层里相同数值只用第一个」，其余同值分支
就整棵剪掉。`i > start` 这一项是关键——它把去重限定在「同一层」（当前这一步的
选择），从而允许不同层选到相同值（比如子集 `[2, 2]` 本来就是合法答案）。

**这一招的统一名字**：排序后「同一层去重」。它不只用于回溯：三数之和（第 1 篇
15 题）里跳过重复也是同一思想，只是那里没用递归。

### 90. 子集 II（中等）

**题目**：给你一个可能包含重复元素的整数数组 `nums`，返回所有不重复的子集。
解集不能包含重复的子集，可以按任意顺序返回。

**思路**：
直接套 78 题的模板会产出重复子集：对 `[2, 2]`，先选第一个 2 还是第二个 2
会得到两个一模一样的 `[2]`。把数组排序后，相同的数相邻，于是在每层循环里
跳过「不是本层第一个、但和本层前一个数相同」的分支即可。

每个节点仍然都要收集（子集从根到任意节点都是答案），这一点和 78 一致。

**代码**（`src/backtracking/subsets_ii.py` / `.cpp`）：

```python
def subsets_with_dup(nums):
    nums = sorted(nums)
    res = []
    path = []

    def backtrack(start):
        res.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res
```

```cpp
void backtrack(const std::vector<int> &nums, int start,
               std::vector<int> &path, std::vector<std::vector<int>> &res) {
    res.push_back(path);
    for (int i = start; i < static_cast<int>(nums.size()); ++i) {
        if (i > start && nums[i] == nums[i - 1]) continue;
        path.push_back(nums[i]);
        backtrack(nums, i + 1, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> subsetsWithDup(std::vector<int> nums) {
    std::sort(nums.begin(), nums.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(nums, 0, path, res);
    return res;
}
```

- **复杂度**：时间 O(n·2^n)，空间 O(n)（递归深度，不含答案本身）。
- **易错点**：忘记先排序，去重判据 `nums[i] == nums[i-1]` 就失效；
  判据写成 `nums[i] == nums[i-1]` 而漏掉 `i > start`，会把 `[2, 2]` 这种
  合法的「不同层重复」也误杀；`res.append(path[:])` 的复制快照不能省。
- **相似题**：78. 子集（无重复版本）、47. 全排列 II（同一去重思想用在排列上）、
  40. 组合总和 II（去重 + 目标和）。

### 47. 全排列 II（中等）

**题目**：给定一个可包含重复数字的序列 `nums`，按任意顺序返回所有不重复的全排列。

**思路**：
在 46 题「`used` 标记已用元素」的基础上加去重。排序后，每层循环里加一条规则：

> 若 `i > 0` 且 `nums[i] == nums[i-1]` 且 `nums[i-1]` 还没被使用，就跳过 `nums[i]`。

**为什么要多一个「前一个相同数未被使用」的条件**：
`used[i-1]` 为真，说明 `nums[i-1]` 正在当前路径的上层（例如已排成 `[1, 1, ...]`），
这时再放 `nums[i]` 是合法且必需的；只有当 `nums[i-1]` 闲置、我们却越过它去选
同值的 `nums[i]` 时，才是在「同一层重复选值」，必须剪掉。换句话说，
这个条件确保相同数值在排列中只能按「从左到右」的顺序被填入各个坑位。

**代码**（`src/backtracking/permutations_ii.py` / `.cpp`）：

```python
def permute_unique(nums):
    nums = sorted(nums)
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
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
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
        if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;
        used[i] = true;
        path.push_back(nums[i]);
        backtrack(nums, path, used, res);
        path.pop_back();
        used[i] = false;
    }
}
```

- **复杂度**：时间 O(n·n!)，空间 O(n)（递归深度 + `used`，不含答案本身）。
- **易错点**：循环仍从 0 开始（排列不限制顺序），不能改成 `start`；
  去重条件里的 `!used[i-1]` 方向容易记反，记住「前一个同值被用了才允许继续」；
  `used[i]` 的复位不能漏。
- **相似题**：46. 全排列（无重复版本）、90. 子集 II（同层去重）、
  40. 组合总和 II（去重 + 目标和）。

---

## 模式四：目标和的组合——可重复选取 vs 每数一次

**适用信号**：题目问「从数组里挑一些数，使它们的和等于目标值，返回所有组合」。
两个变体的分水岭是：**同一个数能否重复选取**，以及**数组是否含重复元素**。

**核心动作**：都用 `start` 只往后选，参数里带一个「还差多少」`remain`。
区别只在两点：
- 可重复选取（39）：下一层传 `i`，本数还能再用；
- 每数只用一次（40）：下一层传 `i + 1`，并加「同层去重」以消除重复组合。

**为什么 `remain` 比「先收集再求和」好**：把目标和当成一路递减的余额，
一旦某数已超过余额（数组有序时后面的更大）就能立即 `break` 剪枝；
余额归零时正好收集，省去每次对整条路径求和的开销。

### 39. 组合总和（中等）

**题目**：给你一个无重复元素的整数数组 `candidates` 和一个目标整数 `target`，
找出所有可以使数字和为目标数 `target` 的不同组合。`candidates` 中的同一个数字
可以无限制重复被选取。

**思路**：
用 `start` 保证组合不重复（`[2, 3]` 与 `[3, 2]` 只算一个），用「下一层传 `i`」
允许重复选取同一个数，用 `remain` 记录还差多少。

**为什么传 `i` 而不是 `i + 1` 就能「可重复」**：`i` 让下一层的起点仍包含当前数，
于是路径里可以连续出现多个相同数字；而起点不回头，又保证不会用不同顺序重复
生成同一组合。这两个约束各管一件事，缺一不可。

**代码**（`src/backtracking/combination_sum.py` / `.cpp`）：

```python
def combination_sum(candidates, target):
    res = []
    path = []

    def backtrack(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])
            path.pop()

    backtrack(0, target)
    return res
```

```cpp
void backtrack(const std::vector<int> &candidates, int start, int remain,
               std::vector<int> &path, std::vector<std::vector<int>> &res) {
    if (remain == 0) {
        res.push_back(path);
        return;
    }
    for (int i = start; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i] > remain) break;
        path.push_back(candidates[i]);
        backtrack(candidates, i, remain - candidates[i], path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combinationSum(std::vector<int> candidates,
                                             int target) {
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(candidates, 0, target, path, res);
    return res;
}
```

- **复杂度**：时间与答案规模相关，最坏指数级；空间 O(target / min(candidates))
  （递归深度，不含答案本身）。
- **易错点**：递归传 `i` 而不是 `i + 1`，否则变成「每数一次」；
  排序后才能用 `if candidates[i] > remain: break`（`break` 依赖有序，
  乱序时只能 `continue`）；收集后要 `return`，避免继续往路径里堆。
- **相似题**：40. 组合总和 II（每数一次 + 去重）、77. 组合（固定长度）、
  1049. 最后一块石头的重量 II（第 13 篇背包，可看作「能否凑出某个和」的判定）。

### 40. 组合总和 II（中等）

**题目**：给定一个候选人编号的集合 `candidates` 和一个目标数 `target`，
找出所有可以使数字和为 `target` 的组合。`candidates` 中的每个数字在每个组合中
只能使用一次。注意解集中不能包含重复的组合。

**思路**：
这道题是 39 与 90 的合体，两把锁缺一不可：
- 每个数只能用一次 → 递归传 `i + 1`（与 39 相反）；
- 候选含重复且组合不能重复 → 排序后用同层去重（与 90 一致）。

只传 `i + 1` 挡不住「值相同、下标不同」造成的重复组合；只去重不传 `i + 1`
又会让同一个元素被用多次。两者搭配才正确。

**代码**（`src/backtracking/combination_sum_ii.py` / `.cpp`）：

```python
def combination_sum2(candidates, target):
    candidates = sorted(candidates)
    res = []
    path = []

    def backtrack(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, remain - candidates[i])
            path.pop()

    backtrack(0, target)
    return res
```

```cpp
void backtrack(const std::vector<int> &candidates, int start, int remain,
               std::vector<int> &path, std::vector<std::vector<int>> &res) {
    if (remain == 0) {
        res.push_back(path);
        return;
    }
    for (int i = start; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i] > remain) break;
        if (i > start && candidates[i] == candidates[i - 1]) continue;
        path.push_back(candidates[i]);
        backtrack(candidates, i + 1, remain - candidates[i], path, res);
        path.pop_back();
    }
}

std::vector<std::vector<int>> combinationSum2(std::vector<int> candidates,
                                              int target) {
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::vector<int>> res;
    std::vector<int> path;
    backtrack(candidates, 0, target, path, res);
    return res;
}
```

- **复杂度**：时间与答案规模相关，最坏指数级；空间 O(n)（递归深度，不含答案本身）。
- **易错点**：递归传 `i + 1`（不是 `i`）；同层去重的 `i > start` 不能漏，
  否则 `[1, 1, 6]` 这类合法组合会被误杀；`break` 依赖排序，记得先 `sort`。
- **相似题**：39. 组合总和（可重复版本）、90. 子集 II（同层去重）、
  416. 分割等和子集（第 13 篇背包，判定能否凑出 `sum/2`）。

---

## 模式五：把决策树铺到网格 / 字符串上

**适用信号**：搜索空间不是一维数组，而是**二维网格**（找路径）或**字符串的切割点**
（切分成若干合法子串）。选择从「选哪个下标」变成「往哪个方向走 / 下一刀切哪」。

**核心动作**：树的结构照旧，只是枚举对象变了。网格题要处理「同一格不能用两次」，
常用**原地改字符再改回来**代替 `visited` 数组；字符串切割题用下标区间
`[start, end)` 表示当前块，只往后切以避免顺序重复。

### 79. 单词搜索（中等）

**题目**：给定一个 `m x n` 的二维字符网格 `board` 和一个字符串单词 `word`。
如果 `word` 存在于网格中，返回 `true`；否则返回 `false`。单词必须按字母顺序
由水平或垂直相邻的格子构成，同一个格子内的字母不允许被重复使用。

**思路**：
把决策树铺在网格上：从任意格子出发，每一步向上下左右试探，只要下一格字母与
单词下一位相同就深入。走完整串即成功。

「不能重复使用同一格」靠一个原地技巧实现：进入格子先把 `board[r][c]` 改成占位符
`'#'` 表示已访问，四个方向递归完再还原。这样无需额外 `visited` 数组。

**为什么最后必须还原**：同一格可能在另一条合法路线里被再次使用（不同起点、
不同拐弯），不还原会把后续搜索错误地挡在门外。这与数组回溯里「撤销选择」是
同一件事，只不过这里撤销的是被改写的棋盘字符。

**代码**（`src/backtracking/word_search.py` / `.cpp`）：

```python
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def dfs(r, c, k):
        if k == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[k]:
            return False
        saved = board[r][c]
        board[r][c] = "#"
        found = (
            dfs(r + 1, c, k + 1)
            or dfs(r - 1, c, k + 1)
            or dfs(r, c + 1, k + 1)
            or dfs(r, c - 1, k + 1)
        )
        board[r][c] = saved
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

```cpp
bool dfs(std::vector<std::string> &board, const std::string &word, int r,
         int c, int k) {
    if (k == static_cast<int>(word.size())) return true;
    int rows = board.size(), cols = board[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != word[k])
        return false;
    char saved = board[r][c];
    board[r][c] = '#';
    bool found = dfs(board, word, r + 1, c, k + 1) ||
                 dfs(board, word, r - 1, c, k + 1) ||
                 dfs(board, word, r, c + 1, k + 1) ||
                 dfs(board, word, r, c - 1, k + 1);
    board[r][c] = saved;
    return found;
}

bool exist(std::vector<std::string> board, const std::string &word) {
    int rows = board.size(), cols = board[0].size();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (dfs(board, word, r, c, 0)) return true;
        }
    }
    return false;
}
```

- **复杂度**：时间 O(m·n·3^L)（每个起点最多向 3 个未回头方向扩展，`L` 为词长），
  空间 O(L)（递归深度）。
- **易错点**：`k == len(word)` 的成功判定要放在越界 / 字符检查**之前**，
  否则匹配到最后一格时下标检查会误报失败；`board[r][c]` 的还原不能漏；
  起点要遍历每个格子，不是只从 `(0, 0)` 出发。
- **相似题**：200. 岛屿数量、130. 被围绕的区域（第 10 篇网格 DFS）、
  212. 单词搜索 II（Trie 剪枝版）。

### 131. 分割回文串（中等）

**题目**：给你一个字符串 `s`，把它分割成一些子串，使每个子串都是回文串。
返回 `s` 所有可能的分割方案。

**思路**：
把「分割」想成在字符串上画竖线：当前块是 `s[start:end]`，如果是回文就作为方案的
一块，然后对后缀从 `end` 继续切。当 `start` 走到串尾，说明切完，收集 `path`。

参数用「起点下标 `start`」而不是「剩余字符串」，好处是天然做到「只往后切」，
既省去反复切片拷贝，也避免顺序不同、内容相同的重复方案。

**为什么用下标区间判定回文**：`is_palindrome(lo, hi)` 在原串上双向收拢，
不产生额外子串对象，比每次 `piece == piece[::-1]` 更省。

**代码**（`src/backtracking/palindrome_partitioning.py` / `.cpp`）：

```python
def partition(s):
    res = []
    path = []

    def is_palindrome(lo, hi):
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1
            hi -= 1
        return True

    def backtrack(start):
        if start == len(s):
            res.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            if not is_palindrome(start, end - 1):
                continue
            path.append(s[start:end])
            backtrack(end)
            path.pop()

    backtrack(0)
    return res
```

```cpp
bool isPalindrome(const std::string &s, int lo, int hi) {
    while (lo < hi) {
        if (s[lo] != s[hi]) return false;
        ++lo;
        --hi;
    }
    return true;
}

void backtrack(const std::string &s, int start, std::vector<std::string> &path,
               std::vector<std::vector<std::string>> &res) {
    if (start == static_cast<int>(s.size())) {
        res.push_back(path);
        return;
    }
    for (int end = start + 1; end <= static_cast<int>(s.size()); ++end) {
        if (!isPalindrome(s, start, end - 1)) continue;
        path.push_back(s.substr(start, end - start));
        backtrack(s, end, path, res);
        path.pop_back();
    }
}

std::vector<std::vector<std::string>> partition(const std::string &s) {
    std::vector<std::vector<std::string>> res;
    std::vector<std::string> path;
    backtrack(s, 0, path, res);
    return res;
}
```

- **复杂度**：时间 O(n·2^n)（最坏约 2^(n-1) 种切法，每次回文判定 O(n)），
  空间 O(n)（递归深度，不含答案本身）。
- **易错点**：`end` 的取值是 `start+1 .. n`（Python `range` 右开所以写 `n+1`，
  C++ 写 `<= n`），别越界；回文判定用 `end - 1` 这个闭区间端点；
  终止条件是 `start == len(s)`，不是 `end`。
- **相似题**：5. 最长回文子串、647. 回文子串（第 13 篇区间 DP，判定回文可预处理）、
  93. 复原 IP 地址（同样是按下标切分字符串）。

---

## 模式六：带约束的构造

**适用信号**：要「构造」出所有满足某些规则的串 / 布局，比如车牌式的字母映射、
合法括号串、棋盘上的皇后。特点是：**可以边构造边校验，非法前缀提前砍掉**，
而不是先造完再筛选。

**核心动作**：把约束翻译成递归参数上的条件（计数、上下界、占用集合），
只有条件成立才继续递归。这样搜索树里根本不会出现非法节点，剪枝天然内建。

### 17. 电话号码的字母组合（中等）

**题目**：给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。
数字到字母的映射与电话按键相同（`2->abc`, `3->def`, …, `9->wxyz`）。

**思路**：
最简单的一棵多叉决策树：从左到右逐位决定这一位选哪个字母，树的深度就是
数字个数。用下标 `i` 表示处理到第几个数字，选完一个字母压入 `path`，
递归 `i+1`，回来弹出；`i` 到末尾时把 `path` 拼成字符串收集。

**为什么不需要 `start` 或 `used`**：每一位数字是一个独立按键，位置天然不同，
既不会有「重复元素」问题，也没有顺序歧义，所以逐位枚举即可——这是回溯最朴素的
形态，正好用来对照后面几个带约束的构造题。

**代码**（`src/backtracking/letter_combinations.py` / `.cpp`）：

```python
def letter_combinations(digits):
    if not digits:
        return []
    table = {
        "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
        "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz",
    }
    res = []
    path = []

    def backtrack(i):
        if i == len(digits):
            res.append("".join(path))
            return
        for ch in table[digits[i]]:
            path.append(ch)
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return res
```

```cpp
void backtrack(const std::string &digits, int i,
               const std::vector<std::string> &table, std::string &path,
               std::vector<std::string> &res) {
    if (i == static_cast<int>(digits.size())) {
        res.push_back(path);
        return;
    }
    for (char ch : table[digits[i] - '0']) {
        path.push_back(ch);
        backtrack(digits, i + 1, table, path, res);
        path.pop_back();
    }
}

std::vector<std::string> letterCombinations(const std::string &digits) {
    if (digits.empty()) return {};
    std::vector<std::string> table = {"",    "",    "abc",  "def", "ghi",
                                      "jkl", "mno", "pqrs", "tuv", "wxyz"};
    std::vector<std::string> res;
    std::string path;
    backtrack(digits, 0, table, path, res);
    return res;
}
```

- **复杂度**：时间 O(4^n·n)（`n` 为数字个数，每位最多 4 个字母，拼接 O(n)），
  空间 O(n)（递归深度，不含答案本身）。
- **易错点**：空输入要单独返回 `[]`（不是 `[""]`）；映射表用字符 `'2'` 作键，
  C++ 里要 `digits[i] - '0'` 转成下标；结束条件用「处理完所有位」`i == len(digits)`。
- **相似题**：22. 括号生成、51. N 皇后（同为逐位 / 逐层构造）、
  78. 子集（同样「每个节点可收集」，但选择来自数组而非按键）。

### 22. 括号生成（中等）

**题目**：数字 `n` 代表生成括号的对数，请设计一个函数，生成所有可能的并且
有效的括号组合。

**思路**：
从左往右逐字符决定放 `(` 还是 `)`。括号串合法的充要条件是：任意前缀里左括号数
不少于右括号数，且最终左右括号数相等。把这两条直接写进递归参数：
- 能放左括号当且仅当 `open < n`；
- 能放右括号当且仅当 `close < open`。

只在满足条件时递归，非法前缀根本不会被生成。

**为什么这样剪枝最划算**：合法性是「前缀性质」——某个前缀一旦非法，往后无论
怎么补都救不回来。所以校验放在每一步做，而不是生成完整串再筛选，能砍掉大量
无效搜索。

**代码**（`src/backtracking/generate_parentheses.py` / `.cpp`）：

```python
def generate_parenthesis(n):
    res = []

    def backtrack(cur, open_, close):
        if len(cur) == 2 * n:
            res.append(cur)
            return
        if open_ < n:
            backtrack(cur + "(", open_ + 1, close)
        if close < open_:
            backtrack(cur + ")", open_, close + 1)

    backtrack("", 0, 0)
    return res
```

```cpp
void backtrack(int n, std::string &cur, int open_, int close,
               std::vector<std::string> &res) {
    if (static_cast<int>(cur.size()) == 2 * n) {
        res.push_back(cur);
        return;
    }
    if (open_ < n) {
        cur.push_back('(');
        backtrack(n, cur, open_ + 1, close, res);
        cur.pop_back();
    }
    if (close < open_) {
        cur.push_back(')');
        backtrack(n, cur, open_, close + 1, res);
        cur.pop_back();
    }
}

std::vector<std::string> generateParenthesis(int n) {
    std::vector<std::string> res;
    std::string cur;
    backtrack(n, cur, 0, 0, res);
    return res;
}
```

- **复杂度**：时间 O(C(2n,n)·n)（结果数为卡特兰数，构造每个串 O(n)），
  空间 O(n)（递归深度，不含答案本身）。
- **易错点**：`close < open_` 的约束不能写成 `close < n`，否则会生成 `)(` 这类
  非法串；终止条件是 `len(cur) == 2*n`；放右括号的条件里比较的是
  `open_`（已用左括号数）而非 `n`。
- **相似题**：17. 电话号码的字母组合（逐位构造）、51. N 皇后（约束满足）、
  20. 有效的括号（第 7 篇，判定一个串是否合法，可与本题对照「判定 vs 生成」）。

### 51. N 皇后（困难）

**题目**：将 `n` 个皇后放在 `n×n` 棋盘上，使皇后彼此不能互相攻击。皇后可以攻击
同一行、同一列或同一斜线上的棋子。返回所有不同的解法，`'Q'` 表示皇后，`'.'`
表示空位。

**思路**：
关键观察：一共 `n` 行、要放 `n` 个皇后，若某行放两个，必有另一行空着，两皇后
迟早冲突。所以**每行恰好一个皇后**，决策树按行展开：第 `r` 层决定这一行的皇后
放哪一列。

每放一个皇后只需检查三样：同列、主对角线（`r - c` 是常量）、副对角线
（`r + c` 是常量）。用三个集合记录已占用的列 / 两条对角线，放子时加入、
回溯时移除，做到 O(1) 冲突检测。行冲突被「每行只放一个」天然排除。

**为什么 `r-c` 与 `r+c` 能代表两条对角线**：同一条主对角线（左上到右下）上任意
两格满足 `r - c` 相等；同一条副对角线（右上到左下）上任意两格满足 `r + c` 相等。
于是「斜线是否冲突」化归为「这两个常量是否出现过」。

**代码**（`src/backtracking/n_queens.py` / `.cpp`）：

```python
def solve_n_queens(n):
    res = []
    board = ["." * n for _ in range(n)]
    cols = set()
    diag_main = set()
    diag_anti = set()

    def backtrack(r):
        if r == n:
            res.append(board[:])
            return
        for c in range(n):
            if c in cols or (r - c) in diag_main or (r + c) in diag_anti:
                continue
            cols.add(c)
            diag_main.add(r - c)
            diag_anti.add(r + c)
            board[r] = board[r][:c] + "Q" + board[r][c + 1:]
            backtrack(r + 1)
            board[r] = board[r][:c] + "." + board[r][c + 1:]
            cols.remove(c)
            diag_main.remove(r - c)
            diag_anti.remove(r + c)

    backtrack(0)
    return res
```

```cpp
void backtrack(int n, int r, std::vector<std::string> &board,
               std::unordered_set<int> &cols, std::unordered_set<int> &diagMain,
               std::unordered_set<int> &diagAnti,
               std::vector<std::vector<std::string>> &res) {
    if (r == n) {
        res.push_back(board);
        return;
    }
    for (int c = 0; c < n; ++c) {
        if (cols.count(c) || diagMain.count(r - c) || diagAnti.count(r + c))
            continue;
        cols.insert(c);
        diagMain.insert(r - c);
        diagAnti.insert(r + c);
        board[r][c] = 'Q';
        backtrack(n, r + 1, board, cols, diagMain, diagAnti, res);
        board[r][c] = '.';
        cols.erase(c);
        diagMain.erase(r - c);
        diagAnti.erase(r + c);
    }
}

std::vector<std::vector<std::string>> solveNQueens(int n) {
    std::vector<std::vector<std::string>> res;
    std::vector<std::string> board(n, std::string(n, '.'));
    std::unordered_set<int> cols, diagMain, diagAnti;
    backtrack(n, 0, board, cols, diagMain, diagAnti, res);
    return res;
}
```

- **复杂度**：时间 O(n!)（每行可选列急剧减少，远小于 n^n），
  空间 O(n)（递归深度 + 三个集合）。
- **易错点**：三个集合插入 / 删除必须成对，漏删会污染兄弟分支；
  `(r-c)` 和 `(r+c)` 别写反成同行判定；棋盘收集要复制快照
  （Python `board[:]`，C++ `push_back(board)` 值拷贝）；`n=1` 时答案是一行 `Q`。
- **相似题**：22. 括号生成（同为约束构造）、37. 解数独、52. N 皇后 II
  （只数解的个数，可去掉棋盘）。

---

## 规律总结

1. **回溯的统一骨架：做选择 → 递归 → 撤销选择**。把问题想成一棵决策树，
   `path` 记录当前路径，循环枚举可选分支，进入下一层前改状态、回来后还原状态。
   「撤销选择」是回溯的灵魂，漏了它兄弟分支就会被污染。状态不一定是数组：
   网格里可以是「改掉的字符」，皇后题里可以是三个占用集合。

2. **先判断「顺序是否有意义」，再决定用 `start` 还是 `used`**。
   - 顺序无关（组合、子集，`[1,2]` 等于 `[2,1]`）→ 用 **`start` 只往后选**，
     强制元素按下标递增出现，天然去重；
   - 顺序有关（排列，`[1,2]` 不同于 `[2,1]`）→ 用 **`used` 标记已用元素**，
     每层从 0 重新挑，只排除用过的。

3. **收集结果的时机取决于题目**：子集在每个节点都收（含根的空集）；
   组合、排列只在路径达到目标长度时收；字符串切割在 `start` 走到末尾时收。
   想清楚「哪些状态算答案」，收集语句的位置就确定了。

4. **含重复元素时，去重靠「排序 + 同层跳过」**（90 / 47 / 40）：
   先 `sort` 让相同的值相邻，再在循环里用「`i > start` 且 `nums[i] == nums[i-1]`」
   跳过同层重复分支；排列（47）因为每层从 0 开始，判据要补一个
   「前一个相同数未被使用」的条件。要点是区分**同层重复**（要剪）和
   **不同层重复**（合法，如子集 `[2,2]`）。

5. **「每个数用一次」还是「可重复用」，只差递归传参**（39 vs 40）：
   传 `i` 表示下一层仍可选当前数（可重复选取），传 `i + 1` 表示本数用过就用完。
   再配合「排序 + 同层去重」处理候选里的重复值，就能覆盖目标和组合的四个变体。
   把目标和写成递减的 `remain`，能在 `remain` 不足时提前 `break` 剪枝。

6. **约束能翻译成参数条件时，就内建成剪枝**（22 / 51）：
   括号题把「左括号数 < n」「右括号数 < 左括号数」写成递归的进入条件；
   皇后题用列 / 两条对角线的占用集合做 O(1) 冲突检测。合法性若是「前缀性质」，
   就该边构造边校验，让非法节点根本不进入搜索树。

7. **剪枝不改变答案，只砍掉注定失败的子树**：77 的循环上界、39/40 的
   `remain` 提前 `break`、22 的括号计数、51 的冲突集合，本质都是「尽早识别
   并放弃」。剪枝是回溯唯一的优化方向（结果集本身的大小是复杂度下界）。

8. **结果的复制不能省**：`path` 是会被反复修改的共享变量，收集时必须
   `res.append(path[:])`（Python）或 `res.push_back(path)`（C++，
   `vector` 是值拷贝）。存引用会让所有答案最终都变成最后一次的状态。

9. **复杂度通常是指数级**：排列 O(n·n!)，子集 O(n·2^n)，组合 O(C(n,k)·k)，
   括号生成是卡特兰数。递归深度多为 O(n) 或 O(L)。拿到回溯题，
   先估「解空间规模」和「每步校验代价」，再谈优化。

10. **和 DFS 是一体两面**：回溯就是「在决策树上做 DFS + 状态撤销」。
    第 10 篇讲的是在图的节点上走，这一篇讲的是在「选择空间」里走，
    骨架（进入 → 递归邻接 → 退出）完全相通；79 题更是把两者合在了一起——
    在网格图上做 DFS，同时用回溯的「改字符再还原」记录路径。
