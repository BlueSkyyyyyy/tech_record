# 并查集：把「连在一起」的关系交给一个结构

刷题时经常遇到这样一类话：「互相连通的」「相等的」「同一组的」「可以互相到达的」。
它们的共同点是**关系会传递**：A 和 B 一组、B 和 C 一组，那么 A 和 C 自然也是一组。
要维护这种「传递性的分组」，最趁手的工具就是**并查集（Disjoint Set Union, DSU）**。

并查集只做两件事，却几乎覆盖了所有「分组 / 连通 / 判环」问题：

- `find(x)`：查 x 属于哪个集合（返回集合的「代表元」，即根节点）；
- `union(a, b)`：把 a、b 所在的两个集合合并成一个。

## 模板先记牢

朴素实现是每个元素指向父亲，一路向上找根；这样最坏会退化成一条链，查询 O(n)。
加两个优化后几乎就是常数时间：

- **路径压缩**：`find` 时顺手把沿途的点直接挂到根上，下次再查一步到位；
- **按秩合并**：合并时把「矮树」挂到「高树」下，避免树越长越高。

两者结合，单次操作的**摊还**复杂度约 O(α(n))（α 是反阿克曼函数，实际不超过 5），
可以当成 O(1)。下面这份模板会贯穿本篇，C++ 与 Python 各一份。

Python（`src/union-find/dsu.py`）：

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
```

C++（每个 `.cpp` 文件顶部都带这一段）：

```cpp
struct DSU {
    std::vector<int> parent, rank_;
    explicit DSU(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    bool unite(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
        return true;
    }
};
```

注意 `union` 的**返回值**：返回 `False` 表示 a、b 原本就同属一个集合——这一位信息在
判环、判矛盾时非常有用，不是可有可无的装饰。

## 本篇要解决的问题与模式

| 模式 | 并查集扮演的角色 | 题目 | 难度 |
|---|---|---|---|
| 模式一：连通分量计数 | 数「分成几块」 | 547. 省份数量 / 765. 情侣牵手 | 中等/困难 |
| 模式二：用返回值判环 | `union` 失败即发现环 | 684. 冗余连接 | 中等 |
| 模式三：等价约束合并后校验 | 先合并相等，再查不等 | 990. 等式方程的可满足性 | 中等 |
| 模式四：把关系抽象成「点」 | 关系本身建图再合并 | 721. 账户合并 / 947. 同行同列石头 / 1202. 交换字符串中的元素 | 中等 |
| 模式五：枚举所有边再合并 | 两两判定 + 计数 | 839. 相似字符串组 | 困难 |
| 模式六：并查集做进阶判定 | 二分图 / 有向图双父 | 785. 判断二分图 / 685. 冗余连接 II | 中等/困难 |

读这一篇的重点是**判断「谁和谁是点、什么关系是边」**。并查集本身很简单，难的是把题面里
缠绕的关系翻译成「点的合并」；一旦翻译对了，代码往往只有十几行。

---

## 模式一：连通分量计数

**适用信号**：题目问「分成几组 / 几块 / 几次就能归位」，而分组关系会传递。

**核心动作**：把每个元素建点，能连的连起来，最后数**不同根的个数**。

### 547. 省份数量（中等）

**题目**：给一个 n×n 的矩阵 `is_connected`，`is_connected[i][j] == 1` 表示城市 i 与 j
直接相连（连通具有传递性）。互相直接或间接连通的城市构成一个「省份」，求省份总数。

**思路**：

把每个城市看成一个点，矩阵里的每条「1」看成一条无向边，省份数就是这张图的
**连通分量个数**。用并查集遍历矩阵，只要 i、j 相连就 `union` 一下，最后数有多少个
不同的根。

**为什么只扫上三角**：矩阵是对称的，i-j 与 j-i 是同一条边，扫一遍就够，扫两遍也不会错，
只是白做。

**代码**（完整可运行版见 `src/union-find/number_of_provinces.py` / `.cpp`）：

```python
from dsu import DSU


def find_circle_num(is_connected):
    n = len(is_connected)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                dsu.union(i, j)
    return len({dsu.find(i) for i in range(n)})
```

```cpp
int findCircleNum(const std::vector<std::vector<int>> &isConnected) {
    int n = isConnected.size();
    DSU dsu(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (isConnected[i][j] == 1) {
                dsu.unite(i, j);
            }
        }
    }
    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return roots.size();
}
```

- **复杂度**：时间近似 O(n²)（主要花在读矩阵），空间 O(n)。
- **易错点**：判连通的是矩阵元素 `== 1`；数分量要看「根的集合」而不是节点的集合，
  写 `sum(dsu.find(i) == i for i in range(n))` 也行（根满足自己指向自己）。
- **相似题**：765 情侣牵手（同为本篇模式一，答案是 `n - 分量数`）；
  200 岛屿数量、547 的 DFS 版本见 `docs/10-graph.md`。

### 765. 情侣牵手（困难）

**题目**：2n 个人坐成一排，第 i 对情侣编号是 2i 与 2i+1。`row` 是当前座位安排（一个排列）。
每次可交换任意两人，求最少交换几次使每对情侣相邻。

**思路**：

情侣 `(2k, 2k+1)` 看作一个「对」，编号 k。观察每对相邻座位 `(2i, 2i+1)`：两个人分属
的对记作 `k1`、`k2`，若 `k1 != k2` 就在 `k1`、`k2` 之间连一条边。

这样得到的图里，一个含 m 个「对」的连通块，只需 **m-1** 次交换就能全部配对
（每次交换把一个对归位、并让错位对数减一）。所以总次数 = Σ(m_j - 1) = n − 连通块数。

**为什么用并查集而不是模拟交换**：我们只关心「最少次数」，而次数只取决于连通块的
大小。并查集不用真的执行交换，直接算出有多少块即可。

**代码**（完整可运行版见 `src/union-find/couples_holding_hands.py` / `.cpp`）：

```python
from dsu import DSU


def min_swaps_couples(row):
    n = len(row) // 2
    dsu = DSU(n)
    for i in range(0, len(row), 2):
        dsu.union(row[i] // 2, row[i + 1] // 2)
    return n - len({dsu.find(i) for i in range(n)})
```

```cpp
int minSwapsCouples(const std::vector<int> &row) {
    int n = row.size() / 2;
    DSU dsu(n);
    for (int i = 0; i < (int)row.size(); i += 2) {
        dsu.unite(row[i] / 2, row[i + 1] / 2);
    }
    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return n - (int)roots.size();
}
```

- **复杂度**：时间 O(n·α(n))，空间 O(n)。
- **易错点**：并查集的元素是「情侣对」而不是「人」，所以要 `row[i] // 2` 换算；
  答案是 `n - 连通块数`，不是连通块数本身；`n` 是座位数的**一半**。
- **相似题**：547 省份数量（同模式，直接数分量）；854 相似度为 K 的字符串（另一套
  思路，非并查集）；1202 交换字符串中的元素（本篇模式四，同为「可交换关系」）。

---

## 模式二：用返回值判环

**适用信号**：无向图里找那条「多余的边」/判断加边后是否成环。

**核心动作**：`union(a, b)` 返回 `False` 当且仅当 a、b 已经连通，此时再加这条边
就会成环。

### 684. 冗余连接（中等）

**题目**：一棵 n 个节点的树多加了一条边，变成恰含一个环的图。给定 n 条无向边
（节点编号 1..n），找出那条可以删掉、使图重新变回树的边；若有多条，返回输入中
最后出现的那条。

**思路**：

逐条处理边 `(a, b)`。若 a、b 此刻还不在同一个连通块，说明这条边安全，`union` 之；
若 a、b 已经连通，那么再加这条边就会成环——它就是答案。

**为什么是「最后出现」**：按输入顺序处理，第一次遇到「两端已连通」的边时，它前面的
边都还没成环且已构成连通块，所以这条就是使环闭合的边。题目保证恰有一个环，
因此第一条冲突边必是答案。

**代码**（完整可运行版见 `src/union-find/redundant_connection.py` / `.cpp`）：

```python
from dsu import DSU


def find_redundant_connection(edges):
    dsu = DSU(len(edges) + 1)
    for a, b in edges:
        if not dsu.union(a, b):
            return [a, b]
    return []
```

```cpp
std::vector<int> findRedundantConnection(const std::vector<std::vector<int>> &edges) {
    DSU dsu(edges.size() + 1);
    for (const auto &e : edges) {
        if (!dsu.unite(e[0], e[1])) {
            return {e[0], e[1]};
        }
    }
    return {};
}
```

- **复杂度**：时间 O(n·α(n))，空间 O(n)。
- **易错点**：节点编号从 1 开始，并查集要开 `n + 1` 个；`union` 的返回值别丢掉，
  判环全靠它；题目保证有且仅有一个环，所以一定能返回。
- **相似题**：685 冗余连接 II（有向图版，多一个「双父」分支，见本篇模式六）；
  128 最长连续序列（另一类「连续关系」，见 `docs/02-hash.md`）；
  261 以图判树（判断 n-1 条边能否构成树，与本题同源）。

---

## 模式三：等价约束合并后校验

**适用信号**：给一堆「相等 / 不等」的约束，问能否同时满足。

**核心动作**：**先把所有「相等」合并，再用「不等」去检查有没有矛盾**。

### 990. 等式方程的可满足性（中等）

**题目**：给定一组形如 `"a==b"` 或 `"a!=b"` 的方程（变量只有小写字母），判断能否给
所有变量赋值使全部方程同时成立。

**思路**：

把「==」看成「在同一集合」，它具有自反、对称、传递性，正好是并查集做的事。分两趟：

1. 第一趟只处理所有 `==`，把相等变量的集合合并；相等有传递性，所以 `a==b`、`b==c`
   会把 a、b、c 合成一个集合；
2. 第二趟处理所有 `!=`，若某个不等式两端竟然落在同一集合，说明前面已推出二者相等，
   矛盾，返回 `False`；全部通过则返回 `True`。

**为什么先合并完所有等式再查不等式**：等式会不断「扩大」连通块，只有等所有等式合并
完毕，集合关系才最终确定。若一边合并一边检查不等式，可能因为合并顺序漏判矛盾。

**代码**（完整可运行版见 `src/union-find/satisfiability_of_equality_equations.py` / `.cpp`）：

```python
from dsu import DSU


def equations_possible(equations):
    dsu = DSU(26)
    for eq in equations:
        if eq[1] == '=':
            dsu.union(ord(eq[0]) - ord('a'), ord(eq[3]) - ord('a'))
    for eq in equations:
        if eq[1] == '!' and dsu.find(ord(eq[0]) - ord('a')) == dsu.find(ord(eq[3]) - ord('a')):
            return False
    return True
```

```cpp
bool equationsPossible(const std::vector<std::string> &equations) {
    DSU dsu(26);
    for (const auto &eq : equations) {
        if (eq[1] == '=') {
            dsu.unite(eq[0] - 'a', eq[3] - 'a');
        }
    }
    for (const auto &eq : equations) {
        if (eq[1] == '!' && dsu.find(eq[0] - 'a') == dsu.find(eq[3] - 'a')) {
            return false;
        }
    }
    return true;
}
```

- **复杂度**：时间 O(n·α(26))，空间 O(26)。
- **易错点**：字符转下标用 `ord(字符) - ord('a')`（C++ 用 `eq[0] - 'a'`）；
  方程格式固定，等号/不等号在 `eq[1]`，两个变量分别在 `eq[0]` 与 `eq[3]`；
  一定要分两趟，不能合并成一趟。
- **相似题**：721 账户合并（本质也是「共享即相等」，见下）、
  399 除法求值（带权并查集，本题的加强版）、1232 缀点成线（斜率相等，非并查集）。

---

## 模式四：把关系抽象成「点」

**适用信号**：题面里没有现成的「节点」，但存在某种「共享 / 同属」关系可以传递。

**核心动作**：先想清楚「谁做点、什么是边」，再套并查集。这一模式最能体现并查集的威力，
因为**建模才是难点**，代码反而最短。

### 721. 账户合并（中等）

**题目**：每个账户是 `[名字, 邮箱1, 邮箱2, ...]`。不同账户只要共享任一个邮箱就是同一个人
（邮箱还会传递合并）。把属于同一个人的账户合并：输出 `[名字, 所有邮箱按字典序排序]`。
结果顺序任意。

**思路**：

名字不能当身份（重名很常见），真正的连接点是**邮箱**——两个账户共享一个邮箱就是
同一个人。步骤：

1. 用哈希表 `email_to_id` 记录每个邮箱「第一次出现在哪个账户」；
2. 遍历每个账户的每个邮箱：若该邮箱之前出现过，说明当前账户与它所属账户是同一个人，
   把两个账户在并查集里 `union`；
3. 按「根账户」把邮箱收拢起来；
4. 每个根账户输出 `[根账户的名字] + 排序后的邮箱列表`。

**为什么哈希表里存的是账户下标而不是邮箱自己**：并查集的元素是「账户」，邮箱只是用来
发现「哪些账户该合并」的线索，它本身不需要进并查集。

**代码**（完整可运行版见 `src/union-find/accounts_merge.py` / `.cpp`）：

```python
from collections import defaultdict

from dsu import DSU


def accounts_merge(accounts):
    email_to_id = {}
    dsu = DSU(len(accounts))
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                dsu.union(i, email_to_id[email])
            else:
                email_to_id[email] = i

    root_to_emails = defaultdict(list)
    for email, i in email_to_id.items():
        root_to_emails[dsu.find(i)].append(email)

    result = []
    for root, emails in root_to_emails.items():
        result.append([accounts[root][0]] + sorted(emails))
    result.sort(key=lambda item: item[1])
    return result
```

```cpp
std::vector<std::vector<std::string>> accountsMerge(
    const std::vector<std::vector<std::string>> &accounts) {
    int n = accounts.size();
    std::unordered_map<std::string, int> email_to_id;
    DSU dsu(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < (int)accounts[i].size(); ++j) {
            const std::string &email = accounts[i][j];
            auto it = email_to_id.find(email);
            if (it != email_to_id.end()) {
                dsu.unite(i, it->second);
            } else {
                email_to_id[email] = i;
            }
        }
    }

    std::map<int, std::vector<std::string>> root_to_emails;
    for (const auto &kv : email_to_id) {
        root_to_emails[dsu.find(kv.second)].push_back(kv.first);
    }

    std::vector<std::vector<std::string>> result;
    for (auto &kv : root_to_emails) {
        std::vector<std::string> emails = kv.second;
        std::sort(emails.begin(), emails.end());
        std::vector<std::string> row;
        row.push_back(accounts[kv.first][0]);
        row.insert(row.end(), emails.begin(), emails.end());
        result.push_back(row);
    }
    std::sort(result.begin(), result.end(),
              [](const std::vector<std::string> &a, const std::vector<std::string> &b) {
                  return a[1] < b[1];
              });
    return result;
}
```

- **复杂度**：时间 O(E·α(A) + E log E)（E 为邮箱总数，A 为账户数，排序占大头），空间 O(E)。
- **易错点**：邮箱要去重（同一账户可能重复列同一邮箱）；名字取自「根账户」`accounts[root][0]`；
  每个账户的邮箱要排序；同一个邮箱在多个账户出现时，后出现的账户要合并到先出现的。
- **相似题**：990 等式方程的可满足性（同为「共享 / 相等」关系）；947 同行同列石头（下题）；
  1202 交换字符串中的元素（下题）。

### 947. 移除最多的同行或同列石头（中等）

**题目**：平面上有若干石头，坐标为 `[x, y]`。若某块石头所在的行或列上还有别的石头，
就可把它移走。求最多能移走多少块。

**思路**：

一块石头架在「行 x」和「列 y」之间。凡是能通过若干石头互相到达的行和列，就属于同一个
连通块；在一个含 k 块石头的连通块里，总能只留下 1 块、移走 **k-1** 块。所以答案是
`n − 连通块个数`。

实现时把行坐标和列坐标分别映射成并查集编号（列编号整体加一个偏移量避免与行冲突），
每块石头 `union(行, 列)`，最后数不同根。

**为什么不同坐标要先离散化**：坐标可能很大也可能为负，直接拿来当下标不安全。用集合
去重 + 字典映射到 `0..m-1` 即可。

**代码**（完整可运行版见 `src/union-find/most_stones_removed_with_same_row_or_column.py` / `.cpp`）：

```python
from dsu import DSU


def remove_stones(stones):
    xs = {x for x, _ in stones}
    ys = {y for _, y in stones}
    x_id = {x: i for i, x in enumerate(xs)}
    y_id = {y: len(xs) + i for i, y in enumerate(ys)}

    dsu = DSU(len(xs) + len(ys))
    for x, y in stones:
        dsu.union(x_id[x], y_id[y])

    roots = {dsu.find(x_id[x]) for x, _ in stones}
    return len(stones) - len(roots)
```

```cpp
int removeStones(const std::vector<std::vector<int>> &stones) {
    std::vector<int> xs, ys;
    for (const auto &s : stones) {
        xs.push_back(s[0]);
        ys.push_back(s[1]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

    std::unordered_map<int, int> x_id, y_id;
    for (int i = 0; i < (int)xs.size(); ++i) x_id[xs[i]] = i;
    for (int i = 0; i < (int)ys.size(); ++i) y_id[ys[i]] = (int)xs.size() + i;

    DSU dsu(xs.size() + ys.size());
    for (const auto &s : stones) {
        dsu.unite(x_id[s[0]], y_id[s[1]]);
    }

    std::unordered_set<int> roots;
    for (const auto &s : stones) roots.insert(dsu.find(x_id[s[0]]));
    return (int)stones.size() - (int)roots.size();
}
```

- **复杂度**：时间 O(n·α(n))，空间 O(n)。
- **易错点**：行列必须用**不同编号**（列加偏移），否则同一行同一列会被误判成同一点；
  每个连通块至少含一个行坐标，所以只统计行节点的根就够；答案是 `n - 分量数`。
- **相似题**：547 省份数量、765 情侣牵手（同属「数分量」一类）；
  721 账户合并（同为「把实体拆成两类元素再连边」的建模）；1135 最低成本联通所有城市（MST）。

### 1202. 交换字符串中的元素（中等）

**题目**：给定字符串 s 和一组可交换的下标对 `pairs`。可以任意多次交换 `pairs` 中任意
一对下标上的字符，求能得到字典序最小的字符串。

**思路**：

交换关系是可传递的：若下标 i 与 j 可换、j 与 k 可换，那么 i、j、k 上的字符可以在它们
之间任意摆放（借助 j 中转）。把下标看成点、`pairs` 看成边，同一个连通分量内的字符可以
随意排列。

于是最优策略：每个连通分量内部，把字符按升序排序，再按下标升序依次填回——字典序最小
就是「小的下标放小的字符」。

**为什么块间互不影响**：不同连通分量之间没有任何可交换路径，字符被锁死在各自块内，
逐块取最优即全局最优。

**代码**（完整可运行版见 `src/union-find/smallest_string_with_swaps.py` / `.cpp`）：

```python
from collections import defaultdict

from dsu import DSU


def smallest_string_with_swaps(s, pairs):
    n = len(s)
    dsu = DSU(n)
    for a, b in pairs:
        dsu.union(a, b)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    result = list(s)
    for indices in groups.values():
        indices.sort()
        chars = sorted(s[i] for i in indices)
        for i, ch in zip(indices, chars):
            result[i] = ch
    return "".join(result)
```

```cpp
std::string smallestStringWithSwaps(std::string s,
                                    const std::vector<std::vector<int>> &pairs) {
    int n = s.size();
    DSU dsu(n);
    for (const auto &p : pairs) {
        dsu.unite(p[0], p[1]);
    }

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[dsu.find(i)].push_back(i);

    for (auto &kv : groups) {
        std::vector<int> indices = kv.second;
        std::sort(indices.begin(), indices.end());
        std::string chars;
        for (int i : indices) chars.push_back(s[i]);
        std::sort(chars.begin(), chars.end());
        for (int k = 0; k < (int)indices.size(); ++k) s[indices[k]] = chars[k];
    }
    return s;
}
```

- **复杂度**：时间 O(n log n)（排序），空间 O(n)。
- **易错点**：块内要把「下标」和「字符」都升序后一一对应，不能只排字符；
  字符串在 Python 里不可变，先转 `list` 再 `join`。
- **相似题**：765 情侣牵手（同为「可交换关系 → 连通块」）；721 账户合并（同为按分量聚合）；
  839 相似字符串组（下题）。

---

## 模式五：枚举所有边再合并

**适用信号**：相似 / 相邻关系没有现成列表，需要两两判定，但数据规模允许平方枚举。

**核心动作**：写一个 `similar(a, b)` 判定函数，双层循环把「边」全部试出来，再套并查集。

### 839. 相似字符串组（困难）

**题目**：若两个字符串可以通过「交换其中恰好两个字符」变得相同（或本来就相同），
就称它们相似。给定一组互为字母异位词的字符串，相似关系可传递，求最终分成几组。

**思路**：

把每个字符串看成一个点。两两检查是否相似，相似就 `union`，最后数连通分量个数。

相似判定：逐位比较，记录不同的位置。不同的位置数为 0（完全相等）或 2，且这两处字符
正好互换（`a[i] == b[j]` 且 `a[j] == b[i]`）——因为题面保证都是字母异位词，「恰好两处
不同」其实就已互换，但显式写出判断更严谨。

**为什么能两两枚举**：n ≤ 300，相似判定是 O(L)，O(n²·L) 足够快；相比用哈希寻找邻居，
直接枚举更简单、不易错。

**代码**（完整可运行版见 `src/union-find/similar_string_groups.py` / `.cpp`）：

```python
from dsu import DSU


def num_similar_groups(strs):
    n = len(strs)
    dsu = DSU(n)

    def similar(a, b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if not diff:
            return True
        if len(diff) != 2:
            return False
        i, j = diff
        return a[i] == b[j] and a[j] == b[i]

    for i in range(n):
        for j in range(i + 1, n):
            if similar(strs[i], strs[j]):
                dsu.union(i, j)
    return len({dsu.find(i) for i in range(n)})
```

```cpp
int numSimilarGroups(const std::vector<std::string> &strs) {
    int n = strs.size();
    DSU dsu(n);

    auto similar = [](const std::string &a, const std::string &b) {
        std::vector<int> diff;
        for (int i = 0; i < (int)a.size(); ++i) {
            if (a[i] != b[i]) diff.push_back(i);
        }
        if (diff.empty()) return true;
        if (diff.size() != 2) return false;
        return a[diff[0]] == b[diff[1]] && a[diff[1]] == b[diff[0]];
    };

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (similar(strs[i], strs[j])) dsu.unite(i, j);
        }
    }

    std::unordered_set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(dsu.find(i));
    return roots.size();
}
```

- **复杂度**：时间 O(n²·L)，空间 O(n)。
- **易错点**：不同的位置只能有 0 个或 2 个，出现 1、3、4… 个都不相似；
  即使两处不同也要确认是「互换」而不是各换各的；相似关系可传递，所以要用并查集而非
  两两计数。
- **相似题**：1202 交换字符串中的元素（同为交换产生的等价类）；
  547 省份数量（数连通分量）；990 等式方程的可满足性（等价关系合并）。

---

## 模式六：并查集做进阶判定

**适用信号**：表面不是「合并分组」题，但核心矛盾可以翻译成「两点意外地同属一组」。

**核心动作**：主动构造一种「同集合 = 矛盾」的判定，或者处理有向图特有的「双父」结构。

### 785. 判断二分图（中等）

**题目**：给定无向图的邻接表 `graph`，判断能否把节点分成两个集合，使每条边的两端都
落在不同集合（即图可以二染色）。

**思路**：

二分图等价于：对任意节点 u，它的所有邻居必须站在 u 的对立面，因此 **u 的所有邻居
彼此必须处在同一阵营**。于是：

- 遍历每个节点 u，把它的所有邻居都 `union` 到一起；
- 若过程中发现 u 和某个邻居已经在同一集合，说明出现了「同色相邻」的边，
  不可能二染色，返回 `False`。

把「同一集合」理解为同色：邻居们都被规定成异于 u 的颜色，即彼此同色；一旦某个邻居和 u
撞进同色集合就矛盾。

**为什么不用 DFS 染色**：DFS 显式给每个点染色并检查冲突，并查集则隐式维护「谁和谁同色」。
两者等价，并查集版本更短，也更能体现「敌人的敌人是朋友」这一关系。

**代码**（完整可运行版见 `src/union-find/is_graph_bipartite.py` / `.cpp`）：

```python
from dsu import DSU


def is_bipartite(graph):
    n = len(graph)
    dsu = DSU(n)
    for u in range(n):
        for v in graph[u]:
            if dsu.find(u) == dsu.find(v):
                return False
            dsu.union(graph[u][0], v)
    return True
```

```cpp
bool isBipartite(const std::vector<std::vector<int>> &graph) {
    int n = graph.size();
    DSU dsu(n);
    for (int u = 0; u < n; ++u) {
        for (int v : graph[u]) {
            if (dsu.find(u) == dsu.find(v)) return false;
            dsu.unite(graph[u][0], v);
        }
    }
    return true;
}
```

- **复杂度**：时间 O(E·α(V))，空间 O(V)。
- **易错点**：`union(graph[u][0], v)` 是把每个邻居 v 与 u 的**首个邻居**合并（代表
  「同色」），不是把 u 和 v 合并；检查冲突要在合并**之前**，否则 u、v 立即同集合。
- **相似题**：684 冗余连接（同用「同集合即矛盾」判环）；785 的 DFS 染色版见
  `docs/10-graph.md`；886 可能的二分法（把「讨厌」当异色边，完全同构）。

### 685. 冗余连接 II（困难）

**题目**：给一个有向图，它由一棵「以某节点为根的有向树」多加一条边得到。找出那条可以
删掉、使图重新变回有向树的边；若有多个答案，返回输入中最后出现的那条。节点编号 1..n。

**思路**：

以某点为根的有向树有两个条件：除根外每个节点**入度为 1**，且无环。多加一条边只会破坏
其中一个（或两个都破坏）：

- 某个节点入度为 2（有两条边指向它），记作「**双父**」；
- 出现一个有向环。

分情况：

1. **没有双父**：问题一定是环，删掉使环闭合的那条边即可，退化成本篇 684；
2. **有双父**（记第二条指向 v 的边为 `conflict`）：答案必是 v 的两条入边之一。
   先假设删掉较晚的 `conflict` 边，看图中还有没有环：
   - 没有环 → 删 `conflict` 即可；
   - 仍有环 → 说明 `conflict` 不是环上的边，必须删掉较早的那条入边。

实现时先照常并查集扫一遍，遇到双父时**不**把 `conflict` 边并入（先把它当作待删），
同时记录是否有环（`union` 失败即环）；最后按上面的规则给出答案。

**代码**（完整可运行版见 `src/union-find/redundant_connection_ii.py` / `.cpp`）：

```python
from dsu import DSU


def find_redundant_directed_connection(edges):
    n = len(edges)
    dsu = DSU(n + 1)
    parent = [0] * (n + 1)
    conflict = -1
    cycle = -1

    for i, (u, v) in enumerate(edges):
        if parent[v] != 0:
            conflict = i
        else:
            parent[v] = u
            if not dsu.union(u, v):
                cycle = i

    if conflict == -1:
        return list(edges[cycle])
    if cycle == -1:
        return list(edges[conflict])
    return [parent[edges[conflict][1]], edges[conflict][1]]
```

```cpp
std::vector<int> findRedundantDirectedConnection(
    const std::vector<std::vector<int>> &edges) {
    int n = edges.size();
    DSU dsu(n + 1);
    std::vector<int> parent(n + 1, 0);
    int conflict = -1, cycle = -1;

    for (int i = 0; i < n; ++i) {
        int u = edges[i][0], v = edges[i][1];
        if (parent[v] != 0) {
            conflict = i;
        } else {
            parent[v] = u;
            if (!dsu.unite(u, v)) cycle = i;
        }
    }

    if (conflict == -1) return edges[cycle];
    if (cycle == -1) return edges[conflict];
    return {parent[edges[conflict][1]], edges[conflict][1]};
}
```

- **复杂度**：时间 O(n·α(n))，空间 O(n)。
- **易错点**：`parent[v]` 记录的是**第一个**父亲（出现双父后不再覆盖），最后要用它
  找较早那条入边；双父时那条边不并入并查集，否则会污染环的判定；无向图不会「双父」，
  这是有向图特有的分支。
- **相似题**：684 冗余连接（无向图版，本题的基础）；207/210 课程表（有向图找环，
  用拓扑排序，见 `docs/10-graph.md`）；1719 重构一棵树的方案数（更复杂的有向树约束）。

---

## 规律总结

1. **并查集只回答一个问题：两个元素是否连通 / 是否同组。** 凡是题面出现
   「传递性的分组、连通、相等、可互换」，就应该先想想并查集。

2. **`union` 的返回值是判环 / 判矛盾的关键。** 返回 `False` 表示两端本就连通：
   无向图里它就是「加这条边会成环」（684）；等价约束里它就是「矛盾」（990）。

3. **建模才是难点：先定「谁做点」。** 并查集代码永远是那十几行，真正决定成败的是把
   题面翻译成点的合并。常见的点有三类：
   - 现成的实体（城市、账户、下标、字符串）；
   - **关系的两类元素**（947 里的「行」和「列」要变成两类点）；
   - **用哈希表把内容映射成点**（721 的邮箱、947 的坐标离散化）。

4. **答案常和「连通分量个数」有关。** 直接数分量（547）；`n - 分量数`（765、947）；
   也可写成「根满足自己指向自己」的计数。

5. **「先合并、后校验」是等价约束的标准节奏。** 等式方程（990）必须先把所有 `==` 合并
   完，再用 `!=` 检查；一边合并一边查会因顺序漏判。

6. **可交换 / 可互换关系 = 连通分量内可自由排列。** 1202 在块内排序回填即最优；
   765 用分量数直接算最少交换次数——都不需要真的执行交换。

7. **并查集不一定只做「合并」，还能做「判定」。** 785 把「邻居必须同色」翻译成
   「邻居彼此同集合」；685 处理有向图特有的「双父 + 环」，把无向图的判环能力延伸过去。

8. **复杂度几乎总是 O(n·α(n)) ≈ O(n)。** 真正的开销常在别处：读矩阵 O(n²)、
   两两相似判定 O(n²)、块内排序 O(n log n)。写复杂度时不要漏掉这些「外围成本」。

9. **路径压缩 + 按秩合并一起用。** 只写路径压缩也能过大部分题，但加上按秩合并才稳妥；
   模板固定，直接背下来即可，不必每次重推。
