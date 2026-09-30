# 二叉树：遍历、递归、层序与构造

二叉树是最能体现「**递归结构**」的数据结构：每个节点本身就是「一个根，加上左子树、右子树」，
而左右子树又是同样的二叉树。这种「自己包含自己」的形状，让绝大多数二叉树题目都能用
**递归**写出极短的代码——只要想清楚两件事：

1. **递归函数对「以某节点为根的子树」做什么**（职责）；
2. **当前节点需要从左右子树的返回值里拿到什么，再往上传什么**（返回值设计）。

本篇从最基础的**三种遍历**与**后序递归的返回值设计**出发，一路走到
**镜像判断、层序遍历、由遍历序列构造、最近公共祖先、验证 BST**——
它们表面各不相同，内核却始终是「递归 + 返回值设计」，或是它的自然延伸（BFS）。

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
| 三种遍历（递归框架） | 144. 二叉树的前序遍历 | 简单 |
| 三种遍历（递归框架） | 94. 二叉树的中序遍历 | 简单 |
| 三种遍历（递归框架） | 145. 二叉树的后序遍历 | 简单 |
| 后序递归：返回值设计 | 104. 二叉树的最大深度 | 简单 |
| 后序递归：返回值设计 | 226. 翻转二叉树 | 简单 |
| 后序递归：返回值设计 | 111. 二叉树的最小深度 | 简单 |
| 后序递归：返回值设计 | 110. 平衡二叉树 | 简单 |
| 后序递归：返回值设计 | 543. 二叉树的直径 | 简单 |
| 后序递归：返回值设计 | 124. 二叉树中的最大路径和 | 困难 |
| 镜像与对称 | 101. 对称二叉树 | 简单 |
| 层序遍历与右视图 | 102. 二叉树的层序遍历 | 中等 |
| 层序遍历与右视图 | 199. 二叉树的右视图 | 中等 |
| 由遍历序列构造 | 105. 从前序与中序构造二叉树 | 中等 |
| 由遍历序列构造 | 106. 从中序与后序构造二叉树 | 中等 |
| 祖先与 BST 校验 | 236. 二叉树的最近公共祖先 | 中等 |
| 祖先与 BST 校验 | 98. 验证二叉搜索树 | 中等 |
| BST 的中序与建树 | 230. 二叉搜索树中第 K 小的元素 | 中等 |
| BST 的中序与建树 | 108. 将有序数组转换为二叉搜索树 | 简单 |
| 原地改造 | 114. 二叉树展开为链表 | 中等 |

---

## 模式一：三种遍历，同一个递归框架

**适用信号**：题目要求按某种顺序「访问」树里的每个节点，或需要在访问时对节点做处理。
关键词常带「前序 / 中序 / 后序 / 遍历 / 按顺序」。

三种遍历的区别，只在于「访问当前节点」这一步插在哪里：

- **前序**：根 → 左 → 右（先记录自己，再进左右子树）；
- **中序**：左 → 根 → 右（先走完左子树，再记录自己，最后走右子树）；
- **后序**：左 → 右 → 根（左右子树都处理完，最后才记录自己）。

它们的递归代码几乎是同一份，只调换三行的顺序。理解了这一点，就不用去背三套模板。

### 144. 二叉树的前序遍历（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的前序遍历（根 → 左 → 右）。

**思路（递归遍历框架）**：
递归函数只有一个职责：把以自己为根的子树按前序填进结果表。到达一个节点时，
先 `append` 自己的值，再递归左子树，最后递归右子树；空节点直接返回。
前序的顺序要求「先根后子」，所以记录自己这一步放在两次递归之前。

为什么递归就够：二叉树的定义本身就是递归的，遍历的顺序也因此天然递归。
写出「访问根、遍历左、遍历右」这三步，树的递归形状会自动把顺序展开成正确的前序序列。
迭代写法要用显式栈来模拟：先压根，每次弹出即记录，再**先压右、后压左**，
这样出栈时才是「左先于右」——不过递归版更短也更不易错，作为主模板即可。

**代码**（完整可运行版见 `src/binary-tree/preorder_traversal.py` / `.cpp`）：

```python
def preorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        result.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return result
```

```cpp
void dfs(TreeNode *node, std::vector<int> &result) {
    if (node == nullptr) return;
    result.push_back(node->val);
    dfs(node->left, result);
    dfs(node->right, result);
}

std::vector<int> preorderTraversal(TreeNode *root) {
    std::vector<int> result;
    dfs(root, result);
    return result;
}
```

- **复杂度**：时间 O(n)（每个节点访问一次），空间 O(h)（h 为树高，即递归栈深度）。
- **易错点**：递归终止条件必须放在最前，且要覆盖 `None`/`nullptr`，否则空子树会继续递归导致崩溃；结果表要定义在递归函数外层并被闭包/引用共享（Python 用闭包变量，C++ 用引用参数），每次递归新建一个表就会丢掉前面的结果；迭代版先压右后压左，顺序写反会得到 根→右→左。
- **相似题**：94. 中序遍历、145. 后序遍历（同一框架，仅调换三行顺序，见下）；589. N 叉树的前序遍历（把两个子节点换成子节点列表的循环）。

### 94. 二叉树的中序遍历（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的中序遍历（左 → 根 → 右）。

**思路（递归遍历框架）**：
和中序只差「访问根」的位置：先递归左子树，再记录自己，最后递归右子树。其余完全一样。

为什么中序特别重要：对**二叉搜索树（BST）**而言，中序遍历的结果一定是
从小到大的有序序列。这让「验证 BST」「找 BST 第 K 小」「求中序后继」等问题，
都能先化归成一次中序遍历。中序 = 有序，是 BST 题里最该记的一条性质。

迭代写法的关键动作是「一路向左压栈」：先把从根开始的整条左链压栈，弹出栈顶
（此刻它没有未访问的左子树）就记录，然后转向它的右子树，重复。这个
「压左链—弹出—转右」的节奏，正是递归调用与返回的显式复刻。

**代码**（`src/binary-tree/inorder_traversal.py` / `.cpp`）：

```python
def inorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        result.append(node.val)
        dfs(node.right)

    dfs(root)
    return result
```

```cpp
void dfs(TreeNode *node, std::vector<int> &result) {
    if (node == nullptr) return;
    dfs(node->left, result);
    result.push_back(node->val);
    dfs(node->right, result);
}

std::vector<int> inorderTraversal(TreeNode *root) {
    std::vector<int> result;
    dfs(root, result);
    return result;
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：`append` 必须夹在两次递归**之间**，放到前面就变成前序、放到后面就变成后序；把中序结果当成「有序表」只在 BST 上成立，普通二叉树不成立，别混用；迭代版别忘了每次「转右」后仍要重新「一路向左」，否则会漏节点。
- **相似题**：98. 验证二叉搜索树、230. BST 第 K 小、501. BST 中的众数（都靠中序遍历；这三题会在后续的树专题展开）；144/145 与本题互为框架变体。

### 145. 二叉树的后序遍历（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的后序遍历（左 → 右 → 根）。

**思路（递归遍历框架）**：
先递归左子树，再递归右子树，最后记录自己。三种遍历共用同一个递归框架，
只是「根」这一步插在不同位置。

后序的独特之处：递归函数**在两个子问题都完成之后**才处理当前节点，
天然适合「需要先知道左右子树的结果，再决定根怎么办」的题目——
求树高、判平衡、算路径和、翻转二叉树都是这个形态。所以后序是「返回值设计」
类题目的默认遍历方式。

迭代写法的常用技巧是「按 根→右→左 走一遍，再把结果整体反转」，就得到 左→右→根；
本篇以递归模板为主。

**代码**（`src/binary-tree/postorder_traversal.py` / `.cpp`）：

```python
def postorder_traversal(root):
    result = []

    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        dfs(node.right)
        result.append(node.val)

    dfs(root)
    return result
```

```cpp
void dfs(TreeNode *node, std::vector<int> &result) {
    if (node == nullptr) return;
    dfs(node->left, result);
    dfs(node->right, result);
    result.push_back(node->val);
}

std::vector<int> postorderTraversal(TreeNode *root) {
    std::vector<int> result;
    dfs(root, result);
    return result;
}
```

- **复杂度**：时间 O(n)，空间 O(h)。
- **易错点**：`append` 放在最后；后序序列的最后一个元素一定是整棵树的根，做「由中序+后序构造树」时靠这条性质定位根；用「先右后左 + 反转」的迭代写法时，要对照 `根→右→左` 而不是 `根→左→右`，否则反转后不是后序。
- **相似题**：106. 从中序与后序遍历序列构造二叉树（利用「后序末位是根」；后续展开）；104. 最大深度、226. 翻转二叉树（都是后序形态，见模式二）；144/94 与本题互为框架变体。

---

## 模式二：后序递归的返回值设计

**适用信号**：题目要求的答案不能只看单个节点，而要**由左右子树的结果汇总**得到。
关键词常带「深度 / 高度 / 路径 / 翻转子树 / 判断是否……」。

这类题的核心不是遍历顺序，而是**递归函数返回什么**。经验规律是：
让递归函数返回「以当前节点为根的子树」对上层有用的那个量（深度、最大路径和、是否平衡……），
上层拿到左右两个返回值后做汇总。因为要先拿到子结果再处理父节点，这类题几乎都是后序。

### 104. 二叉树的最大深度（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的最大深度。最大深度是从根节点到最远叶子节点的最长路径上的节点数。

**思路（后序递归，用返回值汇总左右子树）**：
以 `root` 为根的树，最大深度 = 1（根本身）+ 左右子树中更深的那棵的深度。
这是一个天然的后序递归：先分别问左右子树「你们多深」，等两个答案都回来，
再取较大值加一，作为本层的答案往上返回。空节点深度为 0，是终止条件。

为什么用「返回深度」而不是「传参数累计深度」：深度是由下往上汇总的量——
父节点的答案依赖子节点的答案。把子树深度作为返回值层层上传最自然；
若把当前深度当参数往下传，还要额外维护一个全局最大值去记录见过的最深，反而更绕。
选择「返回值」还是「参数」，是二叉树递归题的核心设计决策。

**代码**（`src/binary-tree/max_depth.py` / `.cpp`）：

```python
def max_depth(root):
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

```cpp
int maxDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    return 1 + std::max(maxDepth(root->left), maxDepth(root->right));
}
```

- **复杂度**：时间 O(n)（每个节点算一次），空间 O(h)。
- **易错点**：空节点返回 `0` 而非 `1`，否则深度会整体多算；「节点数」口径下要 `1 +`，若题目问的是「边数」则不加一，读题要分清；不要用全局变量累加层数，深度必须靠返回值自下而上合并。
- **相似题**：111. 最小深度（对称题，但要注意「单侧为空的节点不是叶子」，不能直接取 `1 + min`）；110. 平衡二叉树（返回深度并在过程中判平衡，见后续）；543. 二叉树的直径（返回深度、顺路更新「经过当前节点的最长路径」）；559. N 叉树的最大深度。

### 226. 翻转二叉树（简单）

**题目**：给定一棵二叉树的根节点 `root`，翻转它并返回根节点。翻转指把每个节点的左右子树互换。

**思路（后序递归，先翻子树再交换）**：
对任意一个节点，翻转后的树 = 左子树翻转后的结果放到右边、右子树翻转后的结果放到左边。
于是递归地：先翻左子树，再翻右子树，最后交换自己的 `left` 和 `right`；空节点直接返回空。

为什么必须先递归再交换：我们要交换的是「两个已经翻转好的子树」。先交换再递归同样能对，
但「后序处理当前节点」的写法与 104、145 一脉相承——当前节点的动作依赖子问题的结果，
用后序最清晰。交换本质是原地修改指针，不需要额外空间。

**代码**（`src/binary-tree/invert_binary_tree.py` / `.cpp`）：

```python
def invert_tree(root):
    if root is None:
        return None
    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

```cpp
TreeNode *invertTree(TreeNode *root) {
    if (root == nullptr) return nullptr;
    root->left = invertTree(root->left);
    root->right = invertTree(root->right);
    std::swap(root->left, root->right);
    return root;
}
```

- **复杂度**：时间 O(n)（每个节点访问一次并交换一次），空间 O(h)。
- **易错点**：交换的是**指针** `left`/`right`，不是值 `val`；Python 的 `a, b = b, a` 会先算右边再赋值，写 `root.left, root.right = invert_tree(root.right), invert_tree(root.left)` 是安全的，但若先给 `root.left` 赋值再用它递归就会拿错子树；空节点要返回 `None`/`nullptr`，否则叶子节点没有终止条件。
- **相似题**：951. 翻转等价二叉树（判断两棵树是否互为翻转）；101. 对称二叉树（比较「左的左 vs 右的右」，本质是镜像，见模式三）；104/145 同属后序形态。

### 111. 二叉树的最小深度（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的最小深度（根节点到最近叶子节点的最短路径上的节点数）。

**思路（后序递归，但要先排除「单侧为空」）**：
直观想法是 `1 + min(左, 右)`，但这是错的。看这棵树：

```
    1
     \
      2
```

根只有右孩子。左子树的深度按定义是 0，若直接取 min 会得到 `1 + 0 = 1`，
可实际根不是叶子，它到最近叶子 2 的距离是 2。错误根源在于：
**空子树不是叶子，不能参与取 min**。

正确做法是先分流：
- 只有左孩子为空 → 答案只能来自右子树，返回 `1 + 右`；
- 只有右孩子为空 → 返回 `1 + 左`；
- 两个孩子都在 → 才返回 `1 + min(左, 右)`；
- 空节点返回 0。

为什么最大深度不用这么麻烦：在 `max` 里 0 永远不可能是最大值（除非两边都空），
空子树不会「污染」结果；而 `min` 会让空子树抢答，所以必须显式排除。

**代码**（`src/binary-tree/min_depth.py` / `.cpp`）：

```python
def min_depth(root):
    if root is None:
        return 0
    if root.left is None:
        return 1 + min_depth(root.right)
    if root.right is None:
        return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))
```

```cpp
int minDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    if (root->left == nullptr) return 1 + minDepth(root->right);
    if (root->right == nullptr) return 1 + minDepth(root->left);
    return 1 + std::min(minDepth(root->left), minDepth(root->right));
}
```

- **复杂度**：时间 O(n)（每个节点一次），空间 O(h)。
- **易错点**：最典型的错法就是直接 `1 + min(左, 右)`，忽略了「单侧为空时另一侧才是有效路径」；注意题目问的是**节点数**，空节点口径与最大深度一致（返回 0）；最小深度是到**叶子**的距离，遇到只有一个孩子的节点不能提前停。
- **相似题**：104. 最大深度（只差 min/max，但最大深度无须分流）；110. 平衡二叉树（也用高度，但判的是差）；543. 二叉树的直径（后续专题）。104/111 对照着记，就能记住「min 要排除空子树」这个坑。

### 110. 平衡二叉树（简单）

**题目**：给定一棵二叉树的根节点 `root`，判断它是否是高度平衡的。高度平衡指：每个节点的左右两棵子树的高度差都不超过 1。

**思路（后序递归：一边求高度，一边判平衡）**：
直观做法是对每个节点调用一次「求高度」，再判断两子树高度差。
但那样每个节点会被重复求高度，最坏 O(n²)。
更好的做法是让递归函数**同时**完成两件事：返回子树高度，顺便检查该子树是否平衡。

约定递归函数返回「高度」，若发现不平衡就返回 -1 作为标记：
- 空节点高度 0；
- 先拿左子树高度，若为 -1 直接返回 -1（短路）；
- 再拿右子树高度，若为 -1 直接返回 -1；
- 若 `|左高 − 右高| > 1`，返回 -1；
- 否则返回 `1 + max(左高, 右高)`。

最终答案就是「返回值不等于 -1」。

为什么能一边求高一边判断：判断一个节点是否平衡，正好需要左右子树的**高度**；
而计算这些高度本来就要递归子节点。把两件事合并到同一次递归里，每个节点只算一次，
复杂度从 O(n²) 降到 O(n)。这是「顺路更新信息」的典型。

**代码**（`src/binary-tree/balanced_binary_tree.py` / `.cpp`）：

```python
def is_balanced(root):
    def height(node):
        if node is None:
            return 0
        left = height(node.left)
        if left == -1:
            return -1
        right = height(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return height(root) != -1
```

```cpp
int height(TreeNode *node) {
    if (node == nullptr) return 0;
    int left = height(node->left);
    if (left == -1) return -1;
    int right = height(node->right);
    if (right == -1) return -1;
    if (std::abs(left - right) > 1) return -1;
    return 1 + std::max(left, right);
}

bool isBalanced(TreeNode *root) {
    return height(root) != -1;
}
```

- **复杂度**：时间 O(n)（每个节点算一次高度），空间 O(h)。
- **易错点**：不要写成「对每个节点各求一次高度」（O(n²)）；-1 是「不平衡」的哨兵，一旦拿到 -1 要立即短路，否则 -1 会被继续当成数值参与 `max`；高度差判断是 `> 1`（等于 1 仍然平衡）；空节点高度是 0。
- **相似题**：104. 最大深度（就是这里的 `height`）；111. 最小深度；543. 二叉树的直径（同样「返回高度、顺路更新全局最优」的模式，见下）；面试常把本题当作「后序 + 剪枝」的入门。

### 543. 二叉树的直径（简单）

**题目**：给定一棵二叉树的根节点 `root`，返回它的直径。直径指树中任意两个节点之间最长路径的**边数**。

**思路（后序递归：返回深度、顺路更新全局最优）**：
一条路径可能不经过根，而是藏在某一棵子树里，所以不能只算「根到最远叶」。
换个视角：任意一条路径，都可以看成「以某个节点为最高点，向左下走一段、向右下走一段」。
这段路径的长度（边数）正好等于**左子树深度 + 右子树深度**。

于是让递归函数 `depth(node)` 返回「以 node 为根的子树深度（节点到最远叶的边数）」，
同时在每个节点处用 `左深 + 右深` 更新答案。父节点拿到的返回值用于继续向上汇总，
而「经过当前节点的最长路径」只用于更新全局最大值，不再往上返回。

为什么要区分「返回值」和「全局量」：向上传递时，路径只能走单侧——父节点只可能接
左边或右边其中一条腿，所以返回值是「单臂长度」；而答案允许左右两条腿都算，
所以要用一个外部变量记录双臂之和的最大值。这是「返回一个量、顺路更新另一个量」的
经典题型，124 最大路径和是它在「带权路径」上的版本。

**代码**（`src/binary-tree/diameter_of_binary_tree.py` / `.cpp`）：

```python
def diameter_of_binary_tree(root):
    best = 0

    def depth(node):
        nonlocal best
        if node is None:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        best = max(best, left + right)
        return 1 + max(left, right)

    depth(root)
    return best
```

```cpp
int depth(TreeNode *node, int &best) {
    if (node == nullptr) return 0;
    int left = depth(node->left, best);
    int right = depth(node->right, best);
    best = std::max(best, left + right);
    return 1 + std::max(left, right);
}

int diameterOfBinaryTree(TreeNode *root) {
    int best = 0;
    depth(root, best);
    return best;
}
```

- **复杂度**：时间 O(n)（每个节点访问一次），空间 O(h)。
- **易错点**：`best` 初值取 0，空树/单节点直径都是 0；更新答案用 `left + right`（边数之和），
  返回值却要 `1 + max(left, right)`，两者别写混；直径的「长度」按**边数**计，若题目改成节点数则要加一；
  路径不一定过根，必须在**每个节点**处都更新，而不是只在根处算一次。
- **相似题**：124. 二叉树中的最大路径和（同骨架，把长度换成和、并处理负数，见下）；
  104. 最大深度（`depth` 就是它）；110. 平衡二叉树（同样返回高度）。

### 124. 二叉树中的最大路径和（困难）

**题目**：给定一棵二叉树的根节点 `root`，返回其任意一条路径的最大路径和。
路径被定义为一条从树中任意节点出发、沿父子边移动、到达任意节点的序列，
同一个节点在路径中最多出现一次（路径不必经过根，也不一定经过叶子）。

**思路（后序递归：返回「单臂最大增益」，顺路更新全局最优）**：
这是 543 的加权版本。定义递归函数 `gain(node)`：以 node 为起点、向下的某一条「单臂」
路径能贡献的最大和。父节点只能接 left 或 right 其中一条腿，所以返回值取两者较大的
那个加上自己。

关键在于**负数处理**：如果某侧子树的最大增益是负数，把这一侧接进来只会让和变小，
不如不接——所以用 `max(gain, 0)` 把它截断成 0（表示「放弃这一侧」）。
这样 `node.val + 左增益 + 右增益` 就是「以 node 为最高点的最佳路径」，用它更新全局答案；
而返回值 `node.val + max(左增益, 右增益)` 只保留单臂继续上传。

为什么全局最优初值要设成负无穷：空子树贡献 0 表示「这里没有节点可选」，但如果整棵树
都是负数（例如 `[-3]`），用 0 截断会得出「一条都不选」的 0，而题目要求路径至少含一个节点。
所以答案初值取负无穷，返回时自然能得到「必选一个节点」的最大值。

**代码**（`src/binary-tree/binary_tree_maximum_path_sum.py` / `.cpp`）：

```python
def max_path_sum(root):
    best = float("-inf")

    def gain(node):
        nonlocal best
        if node is None:
            return 0
        left = max(gain(node.left), 0)
        right = max(gain(node.right), 0)
        best = max(best, node.val + left + right)
        return node.val + max(left, right)

    gain(root)
    return best
```

```cpp
int gain(TreeNode *node, long long &best) {
    if (node == nullptr) return 0;
    int left = std::max(gain(node->left, best), 0);
    int right = std::max(gain(node->right, best), 0);
    long long through = static_cast<long long>(node->val) + left + right;
    if (through > best) best = through;
    return node->val + std::max(left, right);
}

int maxPathSum(TreeNode *root) {
    long long best = LLONG_MIN;
    gain(root, best);
    return static_cast<int>(best);
}
```

- **复杂度**：时间 O(n)（每个节点访问一次），空间 O(h)。
- **易错点**：`best` 初值必须是负无穷，用 0 会在全负树上错成 0；两侧增益要先 `max(..., 0)`
  再相加，负增益不接；返回值只能是「单臂」`node.val + max(left, right)`，不能把左右都带上
  （那样路径就分叉了，父节点接不上）；C++ 里用 `long long` 汇总以防累加溢出。
- **相似题**：543. 二叉树的直径（同一个「返回值 + 全局量」骨架，见上）；
  112. 路径总和、113. 路径总和 II（自顶向下带着剩余和，属另一类）；
  687. 最长同值路径（也是返回单臂、全局取双臂，只是加了「值相同」的约束）。

---

## 模式三：镜像与对称

**适用信号**：题目出现「对称 / 镜像 / 互为翻转 / 两棵树是否相同」，即需要**同时看两个节点**并决定它们的对应关系。这类题不遍历单棵树，而是「双指针式」地在两棵子树上同步递归。

### 101. 对称二叉树（简单）

**题目**：给定一棵二叉树的根节点 `root`，判断它是否轴对称（左右互为镜像）。

**思路（自顶向下比较两个节点，递归）**：
一棵树是否对称，取决于它的左右两棵子树是否互为镜像。
于是问题变成比较两个节点 `a`、`b` 是否镜像：
- 两个都为空 → 镜像是空的，成立；
- 只有一个为空 → 不成立；
- 值不同 → 不成立；
- 值相同 → 继续比较「`a` 的左 vs `b` 的右」和「`a` 的右 vs `b` 的左」。

注意最后一条比较的是**交叉**的两对，而不是同侧的两对——这正是「镜像」的含义。

为什么不能直接比较左右子树相等：「相等」要求 `a` 的左对 `b` 的左、`a` 的右对 `b` 的右；
而「镜像」要求 `a` 的左对 `b` 的右、`a` 的右对 `b` 的左。两者在递归时走的方向不同。

**代码**（`src/binary-tree/symmetric_tree.py` / `.cpp`）：

```python
def is_symmetric(root):
    def is_mirror(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return (
            a.val == b.val
            and is_mirror(a.left, b.right)
            and is_mirror(a.right, b.left)
        )

    return root is None or is_mirror(root.left, root.right)
```

```cpp
bool isMirror(TreeNode *a, TreeNode *b) {
    if (a == nullptr && b == nullptr) return true;
    if (a == nullptr || b == nullptr) return false;
    return a->val == b->val && isMirror(a->left, b->right) &&
           isMirror(a->right, b->left);
}

bool isSymmetric(TreeNode *root) {
    return root == nullptr || isMirror(root->left, root->right);
}
```

- **复杂度**：时间 O(n)（每个节点被比较一次），空间 O(h)。
- **易错点**：交叉比较写成同侧（把 `is_mirror` 退化成「判断两棵树相等」），是本题最常见的错；空节点要先处理，「一个空一个非空」必须返回 `False`；空树/单节点都算对称，边界别漏。
- **相似题**：226. 翻转二叉树（镜像的另一面）；100. 相同的树（把交叉比较改回同侧比较就是它）；951. 翻转等价二叉树。

---

## 模式四：层序遍历与右视图

**适用信号**：题目要求「按层 / 逐层 / 每一层 / 从右侧看」，或答案本身是「每层一个值」。这类题用 **BFS + 队列**，关键技巧是**每轮先数一下当前层有多少个节点**，从而天然分层。

### 102. 二叉树的层序遍历（中等）

**题目**：给定一棵二叉树的根节点 `root`，返回它的层序遍历结果（逐层、从左到右，结果是「列表的列表」）。

**思路（BFS + 队列，逐层切分）**：
层序遍历用队列：先把根入队，然后不断「出队一个节点、把它的左右孩子入队」，
出队顺序自然就是逐层从左到右。难点在于如何把结果**按层分组**。

技巧是每次进入 `while` 循环时先记下 `len(queue)`，这就是当前层的节点数；
然后只出队这么多个节点，作为本层结果收集起来，新入队的孩子属于下一层。
循环一次处理完一整层，把这一层加入结果即可。空树返回空列表。

为什么不能只用一个队列直接铺平输出：题目要的是分层结构（二维）。
先数当前层大小再定长出队，是用 BFS 天然分层的最省事写法；
也可以在处理每个节点时往结果里塞一个「层号」，但那样要多带状态。

**代码**（`src/binary-tree/level_order.py` / `.cpp`）：

```python
from collections import deque


def level_order(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        result.append(level)
    return result
```

```cpp
std::vector<std::vector<int>> levelOrder(TreeNode *root) {
    std::vector<std::vector<int>> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        std::vector<int> level;
        for (int i = 0; i < size; ++i) {
            TreeNode *node = q.front();
            q.pop();
            level.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

- **复杂度**：时间 O(n)（每个节点进出队一次），空间 O(w)（w 为树的最大宽度）。
- **易错点**：`for` 的范围必须用「进入循环时」的 `len(queue)` 快照；若边出队边用变化的长度，分层就会错乱；`deque` 用 `popleft()`（`pop()` 是取右端）；空树要返回空结果而不是 `[[]]`。
- **相似题**：199. 右视图（取每层最后一个）；107. 层序遍历 II（把结果反转）；103. 锯齿形层序遍历（奇数层反转）；429. N 叉树层序遍历。

### 199. 二叉树的右视图（中等）

**题目**：给定一棵二叉树的根节点 `root`，想象自己站在它的右侧，按从顶部到底部的顺序，返回从右侧能看到的节点值。

**思路（层序遍历，取每层最后一个）**：
「从右侧看得到」的节点，正是每一层里最靠右的那个。
于是先做一次 102 的层序遍历，每一层只保留最后一个节点即可。
在按层出队时判断「是不是本层最后一个」（下标等于本层大小减一），是就收集它的值。

为什么用 BFS 而不是 DFS：右视图按「层」定义，BFS 天然按层推进，取每层末元素就是答案，
逻辑最直白。DFS 也能做，比如「先访问右孩子、每个深度只记录第一次遇到的节点」，
但要额外维护深度集合，不如 BFS 直观。

**代码**（`src/binary-tree/right_side_view.py` / `.cpp`）：

```python
from collections import deque


def right_side_view(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            if i == size - 1:
                result.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
    return result
```

```cpp
std::vector<int> rightSideView(TreeNode *root) {
    std::vector<int> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        int size = static_cast<int>(q.size());
        for (int i = 0; i < size; ++i) {
            TreeNode *node = q.front();
            q.pop();
            if (i == size - 1) result.push_back(node->val);
            if (node->left != nullptr) q.push(node->left);
            if (node->right != nullptr) q.push(node->right);
        }
    }
    return result;
}
```

- **复杂度**：时间 O(n)，空间 O(w)。
- **易错点**：判定「最后一个」要用本层固定大小 `size - 1`，不能用当前 `queue` 长度；每层的左右孩子都要入队（不要只入右孩子，否则会漏掉「左孩子撑起下一层」的情况）；空树返回空。
- **相似题**：102. 层序遍历（本题的地基）；116/117. 填充每个节点的下一个右侧节点指针（也是按层维护「右侧」）；637. 二叉树的层平均值（每层一个聚合值）。

---

## 模式五：由遍历序列构造二叉树

**适用信号**：「给定前序/中序/后序中的两种，构造出这棵树」。核心是分治：**用能定根的那一趟定位根，用中序把节点分成左右两堆**，再对两堆递归。配合哈希表把「值 → 中序下标」预存起来，定位根 O(1)。

### 105. 从前序与中序遍历序列构造二叉树（中等）

**题目**：给定一棵树的前序遍历 `preorder` 与中序遍历 `inorder`（无重复元素），构造出这棵二叉树并返回根节点。

**思路（分治：用前序定根、中序分左右）**：
前序的顺序是「根 → 左子树 → 右子树」，所以前序的第一个元素一定是整棵树的根。
中序的顺序是「左子树 → 根 → 右子树」，所以一旦知道根是谁，就能在中序里定位它：
它左边的全部是左子树节点，右边的全部是右子树节点。

拿到左右子树的节点集合后，回到前序里：紧跟根之后、长度为「左子树大小」的一段是左子树的前序，
再往后一段是右子树的前序。于是对两段各自递归，就能还原整棵树。

实现上不真的切数组（那样每次 O(n) 会退化到 O(n²)），而是用「下标区间」表示每段的边界，
再用一个哈希表预先存好「值 → 中序下标」，定位根是 O(1)。这样每个节点只处理一次。

为什么必须知道两种遍历：只有前序无法区分左右子树的分界，只有中序也无法确定谁在上层；
前序给根、中序给分界，两者配合才唯一。106 用「后序定根 + 中序分界」，是同一套思路。

**代码**（`src/binary-tree/construct_from_preorder_inorder.py` / `.cpp`）：

```python
def build_tree_pre_in(preorder, inorder):
    index = {value: i for i, value in enumerate(inorder)}

    def build(pre_lo, pre_hi, in_lo, in_hi):
        if pre_lo > pre_hi:
            return None
        root_val = preorder[pre_lo]
        root = TreeNode(root_val)
        mid = index[root_val]
        left_size = mid - in_lo
        root.left = build(pre_lo + 1, pre_lo + left_size, in_lo, mid - 1)
        root.right = build(pre_lo + left_size + 1, pre_hi, mid + 1, in_hi)
        return root

    return build(0, len(preorder) - 1, 0, len(inorder) - 1)
```

```cpp
TreeNode *buildPreIn(std::vector<int> &preorder, int pre_lo, int pre_hi,
                     int in_lo, int in_hi,
                     std::unordered_map<int, int> &index) {
    if (pre_lo > pre_hi) return nullptr;
    int root_val = preorder[pre_lo];
    TreeNode *root = new TreeNode(root_val);
    int mid = index[root_val];
    int left_size = mid - in_lo;
    root->left = buildPreIn(preorder, pre_lo + 1, pre_lo + left_size, in_lo,
                            mid - 1, index);
    root->right = buildPreIn(preorder, pre_lo + left_size + 1, pre_hi, mid + 1,
                             in_hi, index);
    return root;
}

TreeNode *buildTreePreIn(std::vector<int> preorder, std::vector<int> inorder) {
    std::unordered_map<int, int> index;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i)
        index[inorder[i]] = i;
    return buildPreIn(preorder, 0, static_cast<int>(preorder.size()) - 1, 0,
                      static_cast<int>(inorder.size()) - 1, index);
}
```

- **复杂度**：时间 O(n)（建哈希表 + 每个节点一次），空间 O(n)（哈希表与递归栈）。
- **易错点**：左子树大小是 `mid - in_lo`（区间长度），别写成 `mid`；右子树前序起点是 `pre_lo + left_size + 1`（跳过根和左子树），容易多算/少算 1；终止条件写成 `pre_lo > pre_hi`（空区间），用等号会漏掉叶子；中序有重复元素时本方法不适用。
- **相似题**：106. 从中序与后序构造（后序从末尾取根，见下）；889. 根据前序和后序构造（不唯一，需额外约定）；108. 将有序数组转换为 BST（也靠「取中点作根、左右递归」的分治）。

### 106. 从中序与后序遍历序列构造二叉树（中等）

**题目**：给定一棵树的中序遍历 `inorder` 与后序遍历 `postorder`（无重复元素），构造出这棵二叉树并返回根节点。

**思路（分治：用后序定根、中序分左右）**：
这是 105 的「镜像」版本。后序的顺序是「左子树 → 右子树 → 根」，
所以后序的**最后一个**元素一定是整棵树的根。
知道根后，在中序里定位它：左边是左子树节点、右边是右子树节点。

再回到后序：开头的「左子树大小」个元素是左子树的后序，紧接着的「右子树大小」个是右子树的后序，
最后一位是根。对两段递归即可。同样用「值 → 中序下标」哈希表 + 下标区间，避免切数组，整体 O(n)。

和 105 对照记忆：前序从**前**取根、后序从**后**取根，两者都是「用知根的那一趟定根、用中序分量」，本质同一套分治。

**代码**（`src/binary-tree/construct_from_inorder_postorder.py` / `.cpp`）：

```python
def build_tree_in_post(inorder, postorder):
    index = {value: i for i, value in enumerate(inorder)}

    def build(in_lo, in_hi, post_lo, post_hi):
        if in_lo > in_hi:
            return None
        root_val = postorder[post_hi]
        root = TreeNode(root_val)
        mid = index[root_val]
        left_size = mid - in_lo
        root.left = build(in_lo, mid - 1, post_lo, post_lo + left_size - 1)
        root.right = build(mid + 1, in_hi, post_lo + left_size, post_hi - 1)
        return root

    return build(0, len(inorder) - 1, 0, len(postorder) - 1)
```

```cpp
TreeNode *buildInPost(int in_lo, int in_hi, int post_lo, int post_hi,
                      std::vector<int> &postorder,
                      std::unordered_map<int, int> &index) {
    if (in_lo > in_hi) return nullptr;
    int root_val = postorder[post_hi];
    TreeNode *root = new TreeNode(root_val);
    int mid = index[root_val];
    int left_size = mid - in_lo;
    root->left = buildInPost(in_lo, mid - 1, post_lo, post_lo + left_size - 1,
                             postorder, index);
    root->right = buildInPost(mid + 1, in_hi, post_lo + left_size, post_hi - 1,
                              postorder, index);
    return root;
}

TreeNode *buildTreeInPost(std::vector<int> inorder, std::vector<int> postorder) {
    std::unordered_map<int, int> index;
    for (int i = 0; i < static_cast<int>(inorder.size()); ++i)
        index[inorder[i]] = i;
    return buildInPost(0, static_cast<int>(inorder.size()) - 1, 0,
                       static_cast<int>(postorder.size()) - 1, postorder, index);
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：根是后序的 `post_hi`（末尾）而不是开头；左子树后序右端是 `post_lo + left_size - 1`，右子树后序右端是 `post_hi - 1`（把根让出去），这两个边界最易写错；终止条件同样用 `in_lo > in_hi`；有重复元素时不适用。
- **相似题**：105. 从前序与中序构造（对照记忆）；889. 前序+后序；剑指 Offer 07 重建二叉树（即 105）。

---

## 模式六：最近公共祖先与验证 BST

**适用信号**：一类是「找两个节点的共同祖先 / 它们在哪个子树分叉」（236）；一类是「判断整棵树是否满足某种全局有序约束」（98）。前者是「自下而上汇报找到了谁」，后者是「自顶而下携带约束」，正好是后序与前序的两种典型用法。

### 236. 二叉树的最近公共祖先（中等）

**题目**：给定一棵二叉树和两个节点 `p`、`q`，返回它们的最近公共祖先（LCA）。
最近公共祖先指：所有同时是 `p` 和 `q` 祖先的节点中，深度最大的那个。一个节点也可以是它自己的祖先。

**思路（后序递归，按「子树里找到了谁」返回）**：
定义递归函数 `dfs(node)`：在以 `node` 为根的子树里查找 `p` 和 `q`，返回值含义：
- 若 `node` 为空 → 返回空；
- 若 `node` 就是 `p` 或 `q` → 返回 `node`（找到了其中一个）；
- 否则递归左右子树，拿到 `left`、`right`：
  - `left` 和 `right` 都非空 → 说明 `p`、`q` 分居两侧，`node` 就是它们的 LCA；
  - 只有一侧非空 → 这一侧的结果继续往上返回；
  - 都为空 → 返回空。

为什么这样能得到最近公共祖先：`p`、`q` 要么分居某个节点的左右两侧（该节点就是 LCA），
要么其中一个就是另一个的祖先（此时先遇到的那个直接返回）。递归自下而上把「找到了谁」层层上报，
第一个同时收到两侧结果的节点就是答案。

**代码**（`src/binary-tree/lowest_common_ancestor.py` / `.cpp`）：

```python
def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left is not None and right is not None:
        return root
    return left if left is not None else right
```

```cpp
TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (root == nullptr || root == p || root == q) return root;
    TreeNode *left = lowestCommonAncestor(root->left, p, q);
    TreeNode *right = lowestCommonAncestor(root->right, p, q);
    if (left != nullptr && right != nullptr) return root;
    return left != nullptr ? left : right;
}
```

- **复杂度**：时间 O(n)（每个节点访问一次），空间 O(h)。
- **易错点**：比较节点要用**指针/对象身份**（Python 的 `is`、C++ 的指针相等），不能用 `val`，因为树中值可能重复；「`root` 等于 `p` 或 `q` 就直接返回」这一短路保证了「一个是另一个祖先」时能正确返回上面的那个；`left`/`right` 都非空才返回 `root`，否则返回非空的那一侧。
- **相似题**：235. BST 的最近公共祖先（可利用有序性，从根往下走）；1123. 最深叶节点的最近公共祖先；865. 具有所有最深节点的最小子树；1676. 二叉树的最近公共祖先 IV（多个节点）。

### 98. 验证二叉搜索树（中等）

**题目**：给定一棵二叉树的根节点 `root`，判断它是否是一棵有效的二叉搜索树（BST）。
BST 定义：任意节点的左子树所有值都**小于**它，右子树所有值都**大于**它。

**思路（递归时携带上下界）**：
只比较「父节点和直接孩子」是不够的。反例：

```
      5
     / \
    1   6
       / \
      3   7
```

每个父子关系都满足「左小右大」，但 3 在 5 的右子树里却比 5 小，不是 BST。
问题出在：约束不是来自直接父亲，而是来自**整条祖先链**。

正确做法是递归时带上一个开区间 `(low, high)`：
- 根节点的允许范围是 `(-∞, +∞)`；
- 进入左子树时把上界收紧为当前节点的值，即 `(low, node.val)`；
- 进入右子树时把下界收紧为当前节点的值，即 `(node.val, high)`；
- 当前节点的值必须严格落在 `(low, high)` 内，否则不是 BST。

每个节点只被检查一次，天然满足「祖先链上的所有约束」。

为什么等价于中序有序：中序遍历 BST 会得到严格递增序列，
「值始终落在由祖先收紧的区间内」正是这个递增性的递归表述。

**代码**（`src/binary-tree/validate_bst.py` / `.cpp`）：

```python
def is_valid_bst(root):
    def check(node, low, high):
        if node is None:
            return True
        if not (low < node.val < high):
            return False
        return check(node.left, low, node.val) and check(node.right, node.val, high)

    return check(root, float("-inf"), float("inf"))
```

```cpp
bool checkBst(TreeNode *node, long long low, long long high) {
    if (node == nullptr) return true;
    if (!(low < node->val && node->val < high)) return false;
    return checkBst(node->left, low, node->val) &&
           checkBst(node->right, node->val, high);
}

bool isValidBst(TreeNode *root) {
    return checkBst(root, LLONG_MIN, LLONG_MAX);
}
```

- **复杂度**：时间 O(n)（每个节点一次），空间 O(h)。
- **易错点**：只比较直接父子是经典错误，必须携带祖先约束；边界用开区间（严格不等），出现相等值即不合法；Python 用 `float("-inf"/"inf")` 作初值可避免整数边界问题，C++ 用 `long long` + `LLONG_MIN/LLONG_MAX`（若用 `INT_MIN/MAX`，节点值恰好取到边界时会误判）；中序遍历法也正确，但要额外维护「上一个值」。
- **相似题**：94. 中序遍历（BST 有序性的来源）；700. BST 中的搜索；230. BST 第 K 小（中序计数，见下）；701. BST 中的插入；面试常要求同时给出「上下界法」和「中序法」两种思路。

---

## 模式七：BST 的中序性质与分治建树

**适用信号**：题目里出现「二叉搜索树 / BST」并要找「第 k 小 / 第 k 大 / 某个排名」，
或反过来「给一个有序序列，构造一棵 BST」。前者是「中序 = 有序」的顺用，
后者是它的逆用：**中序有序 ⇒ 取中点作根即可还原平衡 BST**。

### 230. 二叉搜索树中第 K 小的元素（中等）

**题目**：给定一棵二叉搜索树的根节点 `root` 和一个整数 `k`，返回其中第 k 小的元素（k 从 1 开始计数）。

**思路（中序遍历 + 计数，边走边停）**：
二叉搜索树最重要的性质是「中序遍历的结果严格递增」，所以「第 k 小」就等于
「中序遍历序列里的第 k 个元素」。只要做一次中序遍历，用计数器记录已经访问到第几个，
数到第 k 个时直接返回即可。

为什么可以提前停：一旦数到第 k 个，后面更大的元素都不可能是答案，直接返回即可，
不必遍历完整棵树。这比「先完整中序存进数组再取下标 k-1」在 k 很小时更省时间
（虽然后者代码更短、也是常见写法）。

更进阶的写法是给每个节点维护「左子树节点数」，就可以像查排名一样 O(h) 找出答案
（对应 173. BST 迭代器、面试变体）。本篇用中序计数，简单直接。

**代码**（`src/binary-tree/kth_smallest_bst.py` / `.cpp`）：

```python
def kth_smallest(root, k):
    count = 0
    answer = None

    def inorder(node):
        nonlocal count, answer
        if node is None or answer is not None:
            return
        inorder(node.left)
        count += 1
        if count == k:
            answer = node.val
            return
        inorder(node.right)

    inorder(root)
    return answer
```

```cpp
void inorder(TreeNode *node, int k, int &count, int &answer) {
    if (node == nullptr || answer != -1) return;
    inorder(node->left, k, count, answer);
    count += 1;
    if (count == k) {
        answer = node->val;
        return;
    }
    inorder(node->right, k, count, answer);
}

int kthSmallest(TreeNode *root, int k) {
    int count = 0;
    int answer = -1;
    inorder(root, k, count, answer);
    return answer;
}
```

- **复杂度**：时间 O(h + k)（先走到最左下角 O(h)，再访问 k 个），最坏 O(n)；空间 O(h)。
- **易错点**：中序访问顺序是「左 → 根 → 右」，`count` 自增必须夹在两次递归之间；
  计数的口径是从 1 开始数的「第 k 个」，与下标（从 0 开始）差一；找到后要能短路，
  否则后面的节点会把 `answer` 覆盖掉；C++ 里若用 `-1` 当「未找到」哨兵，需确认节点值域不含歧义。
- **相似题**：94. 中序遍历（性质来源）；98. 验证 BST（也用中序有序）；
  173. 二叉搜索树迭代器（用栈实现「中序的暂停与恢复」）；700/701. BST 的搜索与插入。

### 108. 将有序数组转换为二叉搜索树（简单）

**题目**：给定一个按升序排列的整数数组 `nums`，将它转换成一棵**高度平衡**的二叉搜索树，返回它的根节点。（本题有多个合法答案。）

**思路（分治：取中点作根，左右递归）**：
要让 BST 高度平衡，最自然的做法是让左右子树的节点数尽量相等。有序数组的中点正好把
数组分成「比它小的左半」和「比它大的右半」，取中点作根再对两半分别递归，就同时满足了：
左半全小于根、右半全大于根（是 BST），左右规模相差不超过 1（高度平衡）。

这正是「中序序列反推 BST」：只要保证每次取的是当前区间的中间元素作根，还原出的树就平衡。
用下标区间 `[lo, hi]` 递归、不切数组，每个元素只被处理一次。
若取首或尾作根，树会退化成链表（高度 O(n)），这是本题最要避免的。

**代码**（`src/binary-tree/sorted_array_to_bst.py` / `.cpp`）：

```python
def sorted_array_to_bst(nums):
    def build(lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = TreeNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(nums) - 1)
```

```cpp
TreeNode *build(std::vector<int> &nums, int lo, int hi) {
    if (lo > hi) return nullptr;
    int mid = lo + (hi - lo) / 2;
    TreeNode *node = new TreeNode(nums[mid]);
    node->left = build(nums, lo, mid - 1);
    node->right = build(nums, mid + 1, hi);
    return node;
}

TreeNode *sortedArrayToBst(std::vector<int> nums) {
    return build(nums, 0, static_cast<int>(nums.size()) - 1);
}
```

- **复杂度**：时间 O(n)（每个元素建一个节点），空间 O(log n)（平衡树的递归栈深度）。
- **易错点**：终止条件是空区间 `lo > hi`（返回空），用 `lo == hi` 会漏掉单元素区间；
  中点用 `(lo + hi) // 2` 或 `lo + (hi - lo) // 2`，后者可防溢出；左右子树区间是
  `[lo, mid-1]` 与 `[mid+1, hi]`，**中点本身要排除**，否则会重复或死循环；
  空数组要返回空树，测试时别直接对空树解引用。
- **相似题**：109. 有序链表转换 BST（同样的「取中点作根」，只是找中点的成本变高）；
  105/106. 由遍历序列构造（同属「定根 + 分区间」的分治）；
  98. 验证 BST（可用来反向验证结果合法）。

---

## 模式八：原地改造二叉树

**适用信号**：题目要求「原地」把一棵树改成另一种结构（展开成链表、原地转成某种顺序），
不允许新建数组或新树。核心思路是**让递归函数返回改造后子结构的「尾巴 / 端点」**，
这样父节点就能把几段拼起来，而不用重复遍历找接头。

### 114. 二叉树展开为链表（中等）

**题目**：给定一棵二叉树的根节点 `root`，把它原地展开成一个「只有右孩子的链表」，
展开后的顺序与二叉树**前序遍历**的顺序一致，要求原地修改，不额外开数组。

**思路（后序递归：让每棵子树返回它的尾节点）**：
前序顺序是「根 → 左 → 右」。把左子树整条展开后链接到根的右边，再把原来的右子树
接到左链的尾巴上，就完成一次拼接；对每个节点都这样做，最终整棵树就变成前序的右斜链表。

为了知道「左链的尾巴在哪」，递归函数 `dfs(node)` 返回**以 node 为根的子树展开后的最后一个节点**：
- 空节点返回空；
- 先递归展开左右子树，拿到 `left_tail`、`right_tail`；
- 若左子树存在：把右子树挂到左子树尾巴后面（`left_tail.right = node.right`），
  再把整条左链搬到右边（`node.right = node.left`），并清空 `node.left`；
- 返回尾节点：优先 `right_tail`，否则 `left_tail`，都没有就是 `node` 自己。

为什么用后序：拼接动作依赖「左右子树都已经展开好、且知道各自尾巴」，必须先处理子问题
再处理当前节点，正是后序。用返回值传尾节点，避免了为找尾巴再遍历一遍。

另一种等价写法是「逆前序」：按「右 → 左 → 根」的顺序遍历，把每个节点的 `right` 指向
「上一个访问过的节点」，最后访问的（原前序第一个）成为新头。两种写法都是 O(n) 原地。

**代码**（`src/binary-tree/flatten_binary_tree.py` / `.cpp`）：

```python
def flatten(root):
    def dfs(node):
        if node is None:
            return None
        left_tail = dfs(node.left)
        right_tail = dfs(node.right)
        if node.left is not None:
            left_tail.right = node.right
            node.right = node.left
            node.left = None
        return right_tail or left_tail or node

    dfs(root)
```

```cpp
TreeNode *flattenDfs(TreeNode *node) {
    if (node == nullptr) return nullptr;
    TreeNode *left_tail = flattenDfs(node->left);
    TreeNode *right_tail = flattenDfs(node->right);
    if (node->left != nullptr) {
        left_tail->right = node->right;
        node->right = node->left;
        node->left = nullptr;
    }
    if (right_tail != nullptr) return right_tail;
    if (left_tail != nullptr) return left_tail;
    return node;
}

void flatten(TreeNode *root) {
    flattenDfs(root);
}
```

- **复杂度**：时间 O(n)（每个节点访问一次），空间 O(h)（递归栈）。
- **易错点**：拼接顺序不能反——必须**先**把原右子树接到左链尾巴，**再**把左链搬到右边，
  否则 `node.right` 被覆盖后就找不到原右子树了；搬到右边后记得把 `node.left` 置空，
  否则不是「只有右孩子的链表」；返回尾节点时优先 `right_tail`，因为右子树在拼接后排在更后面；
  空节点返回空。Python 里 `right_tail or left_tail or node` 依赖「节点对象为真」，
  节点对象非空即为真，语义正确（不要误以为在用节点值判断）。
- **相似题**：144. 前序遍历（展开顺序就是前序，先写出前序序列再连成链表是朴素解）；
  430. 扁平化多级双向链表（同样的「把子结构插进主链」的拼接思想）；
  116. 填充每个节点的下一个右侧节点指针（也是原地改指针）。

---

## 规律总结

1. **二叉树 = 递归结构，先想递归函数的两件事**：一是它对「以某节点为根的子树」负责做什么；
   二是它返回什么、上层拿这个返回值怎么用。想清这两点，代码通常只有几行。
   写不出递归时，多半是「返回值该是什么」没定下来。

2. **三种遍历是同一个框架**，差别只在「访问当前节点」这一步的位置：
   前序（根左右）放最前，中序（左根右）放中间，后序（左右根）放最后。
   背一份框架、调换三行顺序，比背三套模板划算。

3. **前序适合「带着信息往下走」，后序适合「把结果往上汇总」**。
   需要在进入节点时就决定事情（记录路径、传参），用前序；
   需要等左右子树都算完再决定当前节点（深度、路径和、翻转、判平衡），用后序。
   中序在普通树里多用于「按序输出」，在 BST 里则是「得到有序序列」的代名词。

4. **返回值设计是树题的分水岭**。让递归函数返回上层真正需要的量（深度、和、最优值），
   由父节点合并；若答案不能只靠返回值表达（如「全局最大路径」），就让递归「返回一个量、
   顺路更新另一个全局量」，例如 543 直径、124 最大路径和。**别用全局变量去累计层数**，
   那会把「由下而上汇总」写成「由上而下传递」，既易错又难复用。

5. **空节点是递归的终点，口径要在返回值里统一**。深度题空节点返回 0、
   翻转题空节点返回空、求和题空节点返回 0 或负无穷（视题目允许负值与否而定）。
   终止条件的返回值写错一位，整棵树的结果都会偏。

6. **看清题目的「口径」**：最大深度数的是**节点数**（`1 +`）还是**边数**（不 `+1`）；
   最小深度里「只有一侧孩子的节点不算叶子」；平衡树的「高度差」定义。
   这些细节不涉及复杂算法，却是树题最常丢分的地方。

7. **递归的安全边界**：单次递归的空间是树高 O(h)。树退化成链表时 h = n，
   递归可能栈溢出——这也是很多题提供迭代解的原因。理解递归调用栈的形状
   （进左子树、返回、再进右子树），迭代解和复杂度分析都会顺很多。

8. **「镜像」与「相等」只差比较方向**。判断两棵子树相等，是同侧比同侧；
    判断镜像，是 `a` 的左比 `b` 的右、`a` 的右比 `b` 的左。101 对称二叉树、
    100 相同的树、226 翻转二叉树其实是同一个「双节点递归」框架的三种取值，
    把比较方向想清楚，三题一份模板。

9. **按层处理就用 BFS，技巧是「先数当前层大小」**。进入循环时记下 `len(queue)`，
    只出队这么多节点，本层处理完新入队的自动属于下一层。102 层序遍历取每层全部、
    199 右视图取每层最后一个，都是这个骨架。注意队列的分层靠「出队前快照」，
    而不是靠变化中的长度。递归栈是 O(h)，BFS 队列是 O(w)（最大宽度）。

10. **由两种遍历构造树 = 定根 + 分界 + 区间递归**。能定根的那一趟（前序取首、后序取末）
    给出根，中序负责把节点分成左右两堆，再用「值 → 中序下标」哈希表把定位做到 O(1)，
    用下标区间代替切数组，整体 O(n)。105 与 106 只差「从头还是从尾取根」，
    照着同一份骨架写即可。

11. **全局约束要「携带」、局部结论要「汇报」**。98 验证 BST 的约束来自整条祖先链，
    于是把 `(low, high)` 当参数往下带（前序）；236 最近公共祖先要综合左右子树各自看到了谁，
    于是把「找到的节点」当返回值往上带（后序）。**带参数还是用返回值**，是树题最核心的分叉口。
    另外，涉及具体节点（而非值）时，比较要用对象身份 / 指针，不能用 `val`。

12. **BST 的一切都建立在「中序 = 有序」上**。找第 k 小（230）、验证（98）、求后继（285）、
    迭代器（173）都是这句性质的展开；反过来，给有序序列建树（108/109）就是它的逆用——
    取中点作根即可得平衡 BST。只要看到 BST，先想「中序遍历会得到什么」。

13. **要原地改结构，就让递归返回「端点」**。114 展开为链表时，递归函数返回子结构展开后的
    尾节点，父节点据此把「左链尾」和「右链头」接起来，全程 O(1) 额外空间。
    凡是「把几段子结构拼成一段」的原地改造（改指针、串链表、填 next），都用这个套路：
    **返回值携带接头**，而不是回头再遍历去找。拼接时牢记「先保存、再覆盖」的顺序，
    否则被覆盖的指针会丢掉要接的那一段。

14. **`docs` 与 `src` 必须逐字一致**。题解代码与 `code/leetcode/src/binary-tree/`
    下的实现保持完全一致，以经过自测的 `src` 为准，文档只做粘贴。
    C++ 自测时不要把 `{1, 2}` 这类初值列表直接写进 `assert` 实参——花括号里的逗号会被
    当成宏参数分隔符；先把期望值存进变量再比较。
