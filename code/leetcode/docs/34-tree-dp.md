# 树形 DP：把「子树」当成状态

二叉树篇讲过「后序递归 + 返回值设计」；**树形 DP** 就是它的进阶版：当题目要求
「在整棵树上做最优化 / 计数」时，我们让递归函数返回一组**状态值**，父节点据此做决策。
和线性 DP 一样，它也满足「最优子结构」——父节点的最优解由子树的（若干种）最优解拼出；
区别只是「依赖关系」沿树的父子边展开，所以遍历顺序固定为**后序**（先算孩子，再算自己）。

做树形 DP，只需回答两个问题：

1. **子树需要告诉父节点什么信息**（返回值的含义，可以有多个状态）；
2. **父节点如何用孩子的这些信息做选择**（转移），以及**全局最优在哪个节点处结算**。

约定节点结构（Python 与 C++）：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int v = 0, TreeNode *l = nullptr, TreeNode *r = nullptr)
        : val(v), left(l), right(r) {}
};
```

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 返回「选 / 不选」两状态 | 337. 打家劫舍 III | 中等 |
| 返回「净流量 / 平衡量」 | 979. 在二叉树中分配硬币 | 中等 |
| 返回「单臂长度」，在最高点拼接 | 687. 最长同值路径 | 中等 |
| 返回「两个方向」的状态 | 1372. 二叉树中的最长交错路径 | 中等 |
| 返回「覆盖状态机」 | 968. 监控二叉树 | 困难 |
| 返回「是否 BST + 区间聚合」 | 1373. 二叉搜索子树的最大键值和 | 困难 |
| 返回「子树极值」，在每个节点处结算 | 1026. 节点与其祖先之间的最大差值 | 中等 |
| 一般树（孩子数组）+ 拼两条最长的臂 | 2246. 相邻字符不同的最长路径 | 困难 |

> 一句话记住树形 DP：**后序遍历 → 每个节点返回一组状态 → 用自己的值和孩子的状态做转移。
> 难点永远在「状态到底要几个、分别代表什么」。**

---

## 模式一：返回「选 / 不选」两种状态

**适用信号**：父子节点之间有「互斥 / 联动」约束（选了父亲就不能选儿子之类），
需要同时权衡「选当前节点」与「不选当前节点」两种情形。

**核心动作**：让递归返回 `(选, 不选)` 两个值。这样父节点做决定时，两种情形都有据可依。

### 337. 打家劫舍 III（中等）

**题目**：房屋排成二叉树，不能同时偷「直接相连」的两个节点（父子）。求能偷到的最大金额。

**思路**：递归函数返回 `(rob, not_rob)`：
- `rob`：以当前节点为根的子树，且**偷当前节点**时的最大金额；
- `not_rob`：以当前节点为根的子树，且**不偷当前节点**时的最大金额。

后序拿到左右孩子的两个值后：
- 偷自己 ⇒ 孩子都不能偷：`node.val + 左.not_rob + 右.not_rob`；
- 不偷自己 ⇒ 孩子可偷可不偷，各取最大：`max(左.rob, 左.not_rob) + max(右.rob, 右.not_rob)`。

根节点返回的 `max(rob, not_rob)` 就是答案。

**代码**（`src/tree-dp/house_robber_iii.py` / `.cpp`）：

```python
def rob(root):
    def dfs(node):
        if node is None:
            return (0, 0)
        l_rob, l_not = dfs(node.left)
        r_rob, r_not = dfs(node.right)
        rob_here = node.val + l_not + r_not
        not_here = max(l_rob, l_not) + max(r_rob, r_not)
        return (rob_here, not_here)

    return max(dfs(root))
```

```cpp
std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {0, 0};
    auto [l_rob, l_not] = dfs(node->left);
    auto [r_rob, r_not] = dfs(node->right);
    int rob_here = node->val + l_not + r_not;
    int not_here = std::max(l_rob, l_not) + std::max(r_rob, r_not);
    return {rob_here, not_here};
}

int rob(TreeNode *root) {
    auto [a, b] = dfs(root);
    return std::max(a, b);
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：不要把返回值错写成「单个数字」——偷与不偷必须分开传给父节点，否则父节点在「偷自己」时无法排除孩子被偷的情况；空节点返回 `(0, 0)`。
- **相似题**：198 / 213 打家劫舍（线性、环形版本，见动态规划篇）；树形 DP 的同类「选或不选」还有 1373（见本篇模式六）。

---

## 模式二：返回「净流量 / 平衡量」

**适用信号**：题目要求把某种「资源」在树节点间搬运，使每个节点满足某个本地条件，
求搬运次数 / 代价。典型措辞是「每个节点恰好一个 X」「最少移动次数」。

**核心动作**：把每棵子树看成一个整体，返回它相对目标「多出 / 缺少」多少资源
（记作 `balance`）。跨越「当前节点—孩子」这条边的搬运量，恰好等于该孩子子树
`balance` 的绝对值。

### 979. 在二叉树中分配硬币（中等）

**题目**：树上有 n 个节点、共 n 枚硬币（每节点若干）。每次可把一枚硬币移到相邻节点。
求让每个节点恰好一枚硬币的最少移动次数。

**思路**：递归返回子树 `balance = 子树硬币数 - 子树节点数`：
- `balance > 0`：多余，要向外送；`balance < 0`：缺少，要从外部运进。
- 不管方向，跨过某条边的次数都是该孩子子树 `|balance|`，把它累加即可。

`return node.val + 左.balance + 右.balance - 1`。

**代码**（`src/tree-dp/distribute_coins.py` / `.cpp`）：

```python
def distribute_coins(root):
    moves = 0

    def dfs(node):
        nonlocal moves
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        moves += abs(left) + abs(right)
        return node.val + left + right - 1

    dfs(root)
    return moves
```

```cpp
int moves = 0;

int dfs(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = dfs(node->left);
    int right = dfs(node->right);
    moves += std::abs(left) + std::abs(right);
    return node->val + left + right - 1;
}

int distributeCoins(TreeNode *root) {
    moves = 0;
    dfs(root);
    return moves;
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：`balance` 可正可负，累加时取 `abs`；`return` 里要减去 1（自己消耗一枚硬币的目标）。
- **相似题**：1373（BST 聚合）、另一类「让子树内部自洽、把差额往上抛」的题，如某些「平均分配 / 平衡负载」问题，思路同源。

---

## 模式三：返回「单臂长度」，在最高点拼接

**适用信号**：求树上满足某种「父子连续」性质的**最长路径**。这种路径一定可以看成
「从某个最高点向左右各伸出的一条单链」拼起来。

**核心动作**：递归返回「从当前节点向下能延伸的**单臂**长度」；在**每个节点处**用
「左臂 + 右臂」更新全局最优（因为最高点可能不是根），但返回给父节点的只能是较长的那一条臂。

### 687. 最长同值路径（中等）

**题目**：找最长的路径，路径上每个节点值都相同。路径长度按**边的条数**算。

**思路**：递归返回「从当前节点向下、能延伸的同值单臂边数」。
- 若左孩子值与当前相同：左臂 = `左孩子返回值 + 1`，否则为 0；右臂同理。
- 经过当前节点的最长同值路径 = `左臂 + 右臂`，用它更新全局最优。
- 返回给父节点 `max(左臂, 右臂)`，因为父节点只能从一条边接过来。

**代码**（`src/tree-dp/longest_univalue_path.py` / `.cpp`）：

```python
def longest_univalue_path(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        left_arm = left + 1 if node.left and node.left.val == node.val else 0
        right_arm = right + 1 if node.right and node.right.val == node.val else 0
        best = max(best, left_arm + right_arm)
        return max(left_arm, right_arm)

    dfs(root)
    return best
```

```cpp
int best = 0;

int dfs(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = dfs(node->left);
    int right = dfs(node->right);
    int left_arm = (node->left && node->left->val == node->val) ? left + 1 : 0;
    int right_arm = (node->right && node->right->val == node->val) ? right + 1 : 0;
    best = std::max(best, left_arm + right_arm);
    return std::max(left_arm, right_arm);
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：返回的是 `max(左臂, 右臂)` 而不是 `左臂 + 右臂`——父节点只能接一条臂；长度按边数，所以是 `+1` 不是 `+节点数`。
- **相似题**：124. 二叉树中的最大路径和、543. 二叉树的直径（都是「最高点拼接、返回单臂」，见二叉树篇）；本题只是把「同值」换成了「同色」。

---

## 模式四：返回「两个方向」的状态

**适用信号**：路径除了「连续」，还要求**每一步方向交替**（左右左右……）。此时单臂长度
不够，需要分别记录「下一步向左 / 向右」两种情况。

**核心动作**：递归返回 `(go_left, go_right)`，即以当前节点为起点、第一步分别走左 / 右时的最长长度。
父节点把孩子的「反方向」状态接过来：走左之后必须走右。

### 1372. 二叉树中的最长交错路径（中等）

**题目**：从任一节点向下走，相邻两步方向必须相反，求最长交错路径的边数。

**思路**：`dfs(node)` 返回 `(go_left, go_right)`：
- `go_left = 1 + 左孩子.go_right`（走左后下一步必须走右），左孩子不存在则为 0；
- `go_right = 1 + 右孩子.go_left`。
- 每个节点用 `max(go_left, go_right)` 更新全局最优（最高点未必是根）。

**代码**（`src/tree-dp/longest_zigzag_path.py` / `.cpp`）：

```python
def longest_zigzag(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (0, 0)
        left_go_left, left_go_right = dfs(node.left)
        right_go_left, right_go_right = dfs(node.right)
        go_left = 1 + left_go_right if node.left else 0
        go_right = 1 + right_go_left if node.right else 0
        best = max(best, go_left, go_right)
        return (go_left, go_right)

    dfs(root)
    return best
```

```cpp
std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {0, 0};
    auto [left_go_left, left_go_right] = dfs(node->left);
    auto [right_go_left, right_go_right] = dfs(node->right);
    int go_left = node->left ? 1 + left_go_right : 0;
    int go_right = node->right ? 1 + right_go_left : 0;
    best = std::max(best, std::max(go_left, go_right));
    return {go_left, go_right};
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：接的是孩子的**反方向**状态（走左要接 `go_right`），接反了得到的是「一直同向」的路径；空孩子要单独判，不能直接 `1 + 0`。
- **相似题**：687、124（单臂类）；本题是「下一步方向」也进入状态的最短例子。

---

## 模式五：返回「覆盖状态机」

**适用信号**：一个节点的选择会影响「父 / 子」是否被满足（如覆盖、监视、着色），
且要最小化选择数量。

**核心动作**：给每个节点定义几种**状态**，后序返回状态；父节点根据孩子状态决定自己要不要「行动」。
难点是状态之间不能有遗漏或重叠。

### 968. 监控二叉树（困难）

**题目**：每个节点可装摄像头，能覆盖自己、父节点、直接子节点。求覆盖所有节点的最少摄像头数。

**思路**：后序返回三状态：
- `0`：该节点**未被覆盖**（需要父节点放摄像头救它）；
- `1`：该节点**已被覆盖**，但自己没摄像头（被孩子覆盖）；
- `2`：该节点**装了摄像头**。

拿到左右孩子状态后：
- 只要有孩子是 `0` ⇒ 当前节点必须装摄像头（状态 2，计数 +1）；
- 否则只要有孩子是 `2` ⇒ 当前节点被覆盖（状态 1）；
- 否则两个孩子都是 `1` ⇒ 当前节点暂时未被覆盖，交给父节点（状态 0）。

空节点视为「已覆盖」（状态 1）。最后若根仍是 `0`，根上补一个摄像头。

**代码**（`src/tree-dp/binary_tree_cameras.py` / `.cpp`）：

```python
def min_camera_cover(root):
    cameras = 0

    def dfs(node):
        nonlocal cameras
        if node is None:
            return 1
        left = dfs(node.left)
        right = dfs(node.right)
        if left == 0 or right == 0:
            cameras += 1
            return 2
        if left == 2 or right == 2:
            return 1
        return 0

    if dfs(root) == 0:
        cameras += 1
    return cameras
```

```cpp
int cameras = 0;

// 0 未覆盖 / 1 已覆盖 / 2 装摄像头
int dfs(TreeNode *node) {
    if (node == nullptr) return 1;
    int left = dfs(node->left);
    int right = dfs(node->right);
    if (left == 0 || right == 0) {
        ++cameras;
        return 2;
    }
    if (left == 2 || right == 2) return 1;
    return 0;
}

int minCameraCover(TreeNode *root) {
    cameras = 0;
    if (dfs(root) == 0) ++cameras;
    return cameras;
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：空节点必须返回「已覆盖」（1）而不是「未覆盖」（0），否则每个叶子都会逼父节点装摄像头；「孩子没覆盖」优先级最高（先处理），顺序错了会漏装。
- **相似题**：树的着色 / 最小支配集类问题；状态机思想与线性 DP 的「股票状态机」（动态规划篇）一致，只是载体换成了树。

---

## 模式六：返回「是否 BST + 区间聚合」

**适用信号**：要判断「哪些子树满足某种整体性质」，并在此基础上求一个聚合最优值。

**核心动作**：递归返回一个**结构体 / 元组**，把判断所需的一切都带上去。判断子树是否 BST
需要「左子树最大值 < 根 < 右子树最小值」，所以要把 `(是否BST, 最小值, 最大值, 和)` 一并返回。

### 1373. 二叉搜索子树的最大键值和（困难）

**题目**：找出最大的「二叉搜索子树」，返回其节点值之和（不存在则 0）。

**思路**：后序返回 `(is_bst, min_val, max_val, sum_val)`。当前子树是 BST 当且仅当
左右都是 BST，且 `左.max < node.val < 右.min`。成立就更新全局最大和，并返回合并后的区间与和；
不成立整棵子树作废（`is_bst=False`）。空节点返回 `(True, +∞, -∞, 0)`，使单节点天然合法。

**代码**（`src/tree-dp/max_sum_bst.py` / `.cpp`）：

```python
def max_sum_bst(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (True, float("inf"), float("-inf"), 0)
        l_ok, l_min, l_max, l_sum = dfs(node.left)
        r_ok, r_min, r_max, r_sum = dfs(node.right)
        if l_ok and r_ok and l_max < node.val < r_min:
            total = l_sum + r_sum + node.val
            best = max(best, total)
            return (True, min(l_min, node.val), max(r_max, node.val), total)
        return (False, 0, 0, 0)

    dfs(root)
    return best
```

```cpp
struct Info {
    bool is_bst;
    long long mn;
    long long mx;
    long long sum;
};

long long best = 0;

Info dfs(TreeNode *node) {
    if (node == nullptr) return {true, LLONG_MAX, LLONG_MIN, 0};
    Info l = dfs(node->left);
    Info r = dfs(node->right);
    if (l.is_bst && r.is_bst && l.mx < node->val && node->val < r.mn) {
        long long total = l.sum + r.sum + node->val;
        best = std::max(best, total);
        return {true, std::min(l.mn, (long long)node->val),
                std::max(r.mx, (long long)node->val), total};
    }
    return {false, 0, 0, 0};
}

int maxSumBST(TreeNode *root) {
    best = 0;
    dfs(root);
    return static_cast<int>(best);
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：边界空节点要用 `+∞ / -∞`（C++ 用 `LLONG_MAX / LLONG_MIN`），否则单节点判断会失败；一旦子树不是 BST，向上只能报 `False`，不能继续用它的聚合值。
- **相似题**：98. 验证二叉搜索树（判断单个子树是否 BST，是本题的退化版）；333. 最大 BST 子树（求**节点数**最多，把 `sum` 换成 `size` 即可）。

---

## 模式七：返回「子树极值」，在每个节点处结算

**适用信号**：题目问的是「树上任意一对（祖先，后代）」的某种差值 / 距离，而不是某条特定路径。
这类题不必真的自上而下把祖先传下去——把子树信息自下而上收上来，每个节点天然就是子树里所有
节点的祖先。

**核心动作**：递归返回子树的 `(最小值, 最大值)`。回到当前节点时，用「当前值」与这两个极值做差，
就覆盖了「以当前节点为祖先」的所有配对；取全局最大即可。

### 1026. 节点与其祖先之间的最大差值（中等）

**题目**：给定二叉树，对任意一对「祖先 a、后代 b」，求 `|a.val - b.val|` 的最大值。

**思路（后序返回子树极值）**：
每对祖先—后代都唯一对应「某个节点是另一个节点的祖先」。与其自上而下传祖先，不如自下而上：
让 `dfs` 返回**当前子树的最小值和最大值**，回到当前节点时，它一定是子树里所有节点的祖先，
于是

```
以当前节点为祖先的最大差 = max(|node.val - 子树最小值|, |node.val - 子树最大值|)
```

每个节点处结算一次，取全局最大。为什么这样不漏：任意一对 `(祖先 u, 后代 v)`，递归回到 `u` 时
`v` 还在 `u` 的子树里，`u` 的子树极值一定覆盖了 `v.val`，所以这组差一定被算到。

**代码**（`src/tree-dp/maximum_difference_between_node_and_ancestor.py` / `.cpp`）：

```python
def max_ancestor_diff(root):
    best = 0

    def dfs(node):
        nonlocal best
        if node is None:
            return (float("inf"), float("-inf"))
        l_min, l_max = dfs(node.left)
        r_min, r_max = dfs(node.right)
        low = min(l_min, r_min, node.val)
        high = max(l_max, r_max, node.val)
        best = max(best, abs(node.val - low), abs(node.val - high))
        return (low, high)

    dfs(root)
    return best
```

```cpp
int best = 0;

std::pair<int, int> dfs(TreeNode *node) {
    if (node == nullptr) return {INT_MAX, INT_MIN};
    auto [l_min, l_max] = dfs(node->left);
    auto [r_min, r_max] = dfs(node->right);
    int low = std::min({l_min, r_min, node->val});
    int high = std::max({l_max, r_max, node->val});
    best = std::max({best, std::abs(node->val - low), std::abs(node->val - high)});
    return {low, high};
}

int maxAncestorDiff(TreeNode *root) {
    best = 0;
    dfs(root);
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：空节点要返回 `(+∞, -∞)` 这对「空区间」哨兵，否则会把 0 混进极值；`best` 初值为 0
  即可（差值非负），但 C++ 里 `best` 是全局变量，**每次调用前要清零**。
- **相似题**：1373（同样返回聚合信息，只是聚合的是「是否 BST + 区间」）；124 / 543 的「单臂」
  思路则适合求路径，而不是求任意祖先—后代对。

---

## 模式八：一般树（孩子数组）上拼两条臂

**适用信号**：题目给的不是 `left/right`，而是 `parent` 数组或邻接表；要求的仍是「树上最长路径」。
二叉树的那套 `左臂 + 右臂` 结论照样成立，只是「两条臂」要改成「最长的两条孩子臂」。

**核心动作**：按 `parent` 建出孩子数组，后序 `dfs`。每个节点在孩子里挑**最长的两条合法臂**，
用 `top1 + top2 + 1`（节点数）更新全局最优；返回给父节点的只有 `top1 + 1`。

### 2246. 相邻字符不同的最长路径（困难）

**题目**：给定 `parent[i]`（节点 i 的父节点，`parent[0] = -1`）和每个节点的字符 `s[i]`，
求一条路径，使路径上**相邻节点字符都不同**，返回最长路径的节点数。

**思路（取最长的两条臂）**：
这是「最长同值路径」在一般树上的版本，区别有两个：

1. 孩子不再只有两个，要遍历孩子数组；
2. 长度按**节点数**算，单条臂本身也要把起点节点计入。

`dfs` 返回「以当前节点为起点、向下延伸的最长合法单链节点数」：

```
arm = 1 + max(孩子的 dfs 值，且 s[孩子] != s[当前])
```

返回之前，把当前节点能接的最长两条臂拼起来，得到 `top1 + top2 + 1` 个节点，用它更新全局最优。
只能拼两条——路径在当前节点处最多分叉一次；返回给父节点也只能带一条臂。

**代码**（`src/tree-dp/longest_path_with_different_adjacent_characters.py` / `.cpp`）：

```python
def build_children(parent):
    children = [[] for _ in range(len(parent))]
    for node in range(1, len(parent)):
        children[parent[node]].append(node)
    return children


def longest_path(parent, s):
    children = build_children(parent)
    best = 0

    def dfs(node):
        nonlocal best
        top1 = top2 = 0
        for child in children[node]:
            arm = dfs(child)
            if s[child] == s[node]:
                continue
            if arm > top1:
                top1, top2 = arm, top1
            elif arm > top2:
                top2 = arm
        best = max(best, top1 + top2 + 1)
        return top1 + 1

    dfs(0)
    return best
```

```cpp
int longestPath(const std::vector<int> &parent, const std::string &s) {
    int n = static_cast<int>(parent.size());
    std::vector<std::vector<int>> children(n);
    for (int node = 1; node < n; ++node) children[parent[node]].push_back(node);

    int best = 0;
    std::function<int(int)> dfs = [&](int node) {
        int top1 = 0, top2 = 0;
        for (int child : children[node]) {
            int arm = dfs(child);
            if (s[child] == s[node]) continue;
            if (arm > top1) {
                top2 = top1;
                top1 = arm;
            } else if (arm > top2) {
                top2 = arm;
            }
        }
        best = std::max(best, top1 + top2 + 1);
        return top1 + 1;
    };
    dfs(0);
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(n)（孩子数组 O(n)，递归栈 O(h)）。
- **易错点**：`top1`、`top2` 的更新顺序不能写反（先用旧 `top1` 顶替 `top2`）；不合法的手臂（字符相同）
  要直接跳过，不能参与比较；答案按节点数算，所以是 `top1 + top2 + 1`，返回时是 `top1 + 1`。
- **相似题**：124. 二叉树中的最大路径和、543. 二叉树的直径、687. 最长同值路径（同一「两臂拼接」
  骨架，只是载体是二叉树）；1372 把「方向」也放进了状态。

---

## 规律总结

1. **树形 DP = 后序遍历 + 返回值设计**。先算左右孩子，再用它们的结果算自己。递归函数
   返回的「状态」就是本题的 DP 状态，状态定义错了，转移一定写不对。
2. **状态数量由约束决定**：父子互斥 → 返回「选 / 不选」（337）；资源搬运 → 返回「余额」（979）；
   最长路径 → 返回「单臂」（687）或「双向状态」（1372）；覆盖 / 支配 → 返回「状态机」（968）；
   整树性质判断 → 返回「判断所需的全套聚合信息」（1373）；求「祖先—后代」的极值 → 返回子树
   极值（1026）。
3. **全局最优常在「某个节点处结算」**，而不是根节点。路径类（687、1372）必须每个节点都更新答案，
   因为最高点可能在任意位置。
4. **返回给父节点的信息要「够用且不过量」**：路径类不能把左右臂之和传上去（父节点接不了两条臂），
   只能传 `max`；而 BST 判断必须把 min/max 都传上去，少一个就判不了。
5. 空节点的返回值要单独想清楚（`(0,0)` / `+∞,-∞` / 「已覆盖」），它是很多边界 bug 的源头。
6. 树形 DP 的复杂度几乎都是 **O(n) 时间**：每个节点只算一次。若写成对每个节点再扫一遍子树，
   就退化成 O(n²)，那通常意味着返回值设计得不对。
7. **「祖先—后代」类问题不必真的往下传信息**。把子树的最小 / 最大值收上来，每个节点都能当一次
   「祖先」结算答案（1026），比自顶向下带参数更省事。选哪种写法只看一件事：所需信息能不能
   从子树汇总出来——能，就用后序。
8. **孩子多于两个时，模式照样成立**。二叉树里的「左臂 + 右臂」换成「最长的两条孩子臂」
   （2246），核心约束不变：返回给父节点的只能有一条臂，分叉只在当前节点处结算一次。

> 延伸阅读：线性 / 序列 / 背包 / 区间 / 状态机 DP 见「动态规划（一）」；
> 树的基础遍历与后序返回值见「二叉树」。
