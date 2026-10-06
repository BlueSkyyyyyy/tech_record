# 最短路径：选对算法的那一步

**最短路**要回答的问题很朴素：从起点到终点，怎样走代价最小。难点不在「搜索」，而在
**先看清这张图的边有什么性质**——边权是不是都为 1？有没有 0？有没有负数？要不要同时
优化两个量（比如「最便宜」但又有「中转次数」上限）？这些性质决定了该用哪一种最短路算法。

本篇把最短路家族按「边的性质」串成一条线，用十道题覆盖六种模式：

| 模式 | 边/限制的性质 | 算法 | 题目 | 难度 |
|---|---|---|---|---|
| 模式一：边权全为 1 | 每步代价相同 | 普通 BFS | 1091. 二进制矩阵中的最短路径 / 433. 最小基因变化 | 中等 |
| 模式二：边权非负 | 代价可不同 | Dijkstra（优先队列） | 743. 网络延迟时间 / 1631. 最小体力消耗路径 / 1976. 到达目的地的方案数 | 中等 |
| 模式三：边权只有 0/1 | 代价非 0 即 1 | 0-1 BFS（双端队列） | 1368. 使网格图至少有一条有效路径的最小代价 | 困难 |
| 模式四：边数/时间有限制 | 「最多 k 步」类 | Bellman-Ford（按轮松弛） | 787. K 站中转内最便宜的航班 / 1928. 规定时间内到达终点的最小花费 | 中等/困难 |
| 模式五：要所有点对的距离 | 全源最短路 | Floyd-Warshall | 1334. 阈值距离内邻居最少的城市 | 中等 |
| 模式六：瓶颈型「最早时刻」 | 答案单调、可判定 | 二分答案 + BFS | 778. 水位上升的泳池中游泳 | 困难 |

读这一篇的重点是**判断边权和限制属于哪一类**。一旦归类正确，模板几乎可以直接套用；
归错类（例如给无权图硬套 Dijkstra）虽然可能碰巧过，但会白写很多代码。

---

## 模式一：边权全为 1 —— 普通 BFS

**适用信号**：每条边的代价都一样（走一步算一步），求「最少步数 / 最少操作次数」。

**核心动作**：用队列一层层向外扩，第一次到达某点时的层数就是它到起点的最短距离。
不需要优先队列，普通 FIFO 队列就够——因为按层出队天然就是按距离从小到大。

### 1091. 二进制矩阵中的最短路径（中等）

**题目**：给一个 n×n 的 0/1 矩阵，0 是空地、1 是障碍。从左上角到右下角，每步可走
上、下、左、右、四个斜角共 8 个相邻的 0 格，求最短路径经过的格子数；无路返回 -1。

**思路**：

这是最标准的无权最短路：每一步的代价都是 1，用 BFS 逐层扩展即可。`dist[r][c]` 记录
「从起点走到这里经过的格子数」，起点记为 1（起点本身算一格）。八个方向里只走 0 格、
且 `dist` 仍为 0（未访问）的格子，把它标成 `dist[r][c] + 1` 并入队。

**为什么 BFS 就能求最短路**：队列先进先出，距离为 d 的点一定在距离为 d+1 的点之前
出队。因此某个格子第一次被访问时，用的就是最少步数——不会存在「绕远路先到、之后又
发现更短」的情况。这就是无权图不需要 Dijkstra 的原因。

**代码**（完整可运行版见 `src/shortest-path/shortest_path_binary_matrix.py` / `.cpp`）：

```python
from collections import deque

_DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def shortest_path_binary_matrix(grid):
    n = len(grid)
    if grid[0][0] == 1 or grid[n - 1][n - 1] == 1:
        return -1
    dist = [[0] * n for _ in range(n)]
    dist[0][0] = 1
    queue = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        if r == n - 1 and c == n - 1:
            return dist[r][c]
        for dr, dc in _DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and dist[nr][nc] == 0:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return -1
```

```cpp
int shortestPathBinaryMatrix(const std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
        return -1;
    }
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, 0));
    dist[0][0] = 1;
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});

    const int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        if (r == n - 1 && c == n - 1) {
            return dist[r][c];
        }
        for (int d = 0; d < 8; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] == 0 &&
                dist[nr][nc] == 0) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    return -1;
}
```

- **复杂度**：时间 O(n²)（每格至多入队一次），空间 O(n²)。
- **易错点**：起点或终点是障碍要提前返回 -1；用 `dist == 0` 兼作「未访问」标记，所以
  起点必须写成 1 而不是 0；别把「经过格子数」和「走的步数」搞混（本题要的是格子数，
  答案比步数多 1）。
- **相似题**：542. 01 矩阵、994. 腐烂的橘子（`docs/10-graph.md` 的多源 BFS）；
  433 最小基因变化（同为本篇模式一）；1926 迷宫离入口最近的出口（同网格 BFS）。

### 433. 最小基因变化（中等）

**题目**：基因串由 `A/C/G/T` 组成、长度固定为 8。一次「变化」是把某一位换成另一个
字符，且变化后的串必须在基因库 `bank` 中。给定起点串和目标串，求最少变化次数，做不到
返回 -1。

**思路**：

把每个基因串看成一个节点，只差一个字符且都在 `bank` 里的两个串之间连一条边。一次变化
走一条边，于是问题变成「起点到终点的最短路边数」，边权全为 1，又回到 BFS。

这类图的节点不是现成的列表，而是**访问到某个串时才现算它的邻居**：枚举 8 个位置 × 3 个
替换字符，得到至多 24 个候选串，落在 `bank` 里的才算真邻居。这叫**隐式图**——图很大
但结构规整，不必显式建边。

**代码**（`src/shortest-path/min_genetic_mutation.py` / `.cpp`）：

```python
from collections import deque

_GENES = "ACGT"


def min_genetic_mutation(start_gene, end_gene, bank):
    bank = set(bank)
    if end_gene not in bank:
        return -1
    if start_gene == end_gene:
        return 0
    queue = deque([start_gene])
    visited = {start_gene}
    steps = 0
    while queue:
        steps += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            for i in range(len(cur)):
                for g in _GENES:
                    if g == cur[i]:
                        continue
                    nxt = cur[:i] + g + cur[i + 1:]
                    if nxt in bank and nxt not in visited:
                        if nxt == end_gene:
                            return steps
                        visited.add(nxt)
                        queue.append(nxt)
    return -1
```

```cpp
int minGeneticMutation(const std::string &startGene, const std::string &endGene,
                       const std::vector<std::string> &bank) {
    std::unordered_set<std::string> genes(bank.begin(), bank.end());
    if (genes.find(endGene) == genes.end()) {
        return -1;
    }
    if (startGene == endGene) {
        return 0;
    }
    const std::string bases = "ACGT";
    std::queue<std::string> q;
    std::unordered_set<std::string> visited;
    q.push(startGene);
    visited.insert(startGene);
    int steps = 0;
    while (!q.empty()) {
        ++steps;
        int level = q.size();
        for (int i = 0; i < level; ++i) {
            std::string cur = q.front();
            q.pop();
            for (int p = 0; p < static_cast<int>(cur.size()); ++p) {
                char old = cur[p];
                for (char g : bases) {
                    if (g == old) {
                        continue;
                    }
                    cur[p] = g;
                    if (genes.count(cur) && !visited.count(cur)) {
                        if (cur == endGene) {
                            return steps;
                        }
                        visited.insert(cur);
                        q.push(cur);
                    }
                }
                cur[p] = old;
            }
        }
    }
    return -1;
}
```

- **复杂度**：时间 O(L·4·N)（L=8 为串长，N 为基因库大小），空间 O(N)。
- **易错点**：目标不在 `bank` 里要直接 -1（除非起点就是终点）；`visited` 要在**入队时**
  就标记（而不是出队时），否则同一个串会被重复入队；生成邻居时用 `for _ in range(len(queue))`
  按层处理，层数才是变化次数；C++ 里改字符后要还原 `cur[p] = old`。
- **相似题**：127. 单词接龙（同一个「只差一个字符」的隐式图 BFS）；752. 打开转盘锁
  （另一种隐式图）；本篇 1091（都是边权为 1 的 BFS）。

---

## 模式二：边权非负 —— Dijkstra（优先队列）

**适用信号**：各条路的代价不一样（有长有短），但都非负，求最小总代价。

**核心动作**：用小顶堆每次取出「当前已知距离最小」的点来扩展。它一旦出堆，距离就
确定为最优；再用它去松弛邻居。把 BFS 的 FIFO 队列换成优先队列，就得到 Dijkstra。

### 743. 网络延迟时间（中等）

**题目**：n 个节点，有向边 `[u, v, w]` 表示信号从 u 到 v 耗时 w。从节点 k 发信号，
求所有节点都收到信号的最短时间；有节点收不到返回 -1。

**思路**：

所有边权非负，是单源最短路的标准场景，用 Dijkstra 求 k 到每个点的最短距离。因为信号
沿各自最短路径走，**最后一个节点收到的时刻**就是所有最短距离里的最大值；若有点的
距离仍是无穷，说明不可达，返回 -1。

**为什么每次取最小距离的点就对了**：设当前堆顶是 u，距离 d。想找一条比 d 更短的路到
u，就必须先到一个还没确定的点 x、再走到 u；但 x 的距离不小于 d（堆顶最小），而边权
非负，绕过去只会更远。所以 d 已经是 u 的最优解。这个「贪心 + 非负边权」就是 Dijkstra
的正确性来源。

**代码**（`src/shortest-path/network_delay_time.py` / `.cpp`）：

```python
import heapq
from collections import defaultdict


def network_delay_time(times, n, k):
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    INF = float("inf")
    dist = [INF] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    ans = max(dist[1:])
    return -1 if ans == INF else ans
```

```cpp
int networkDelayTime(const std::vector<std::vector<int>> &times, int n, int k) {
    std::vector<std::vector<std::pair<int, int>>> graph(n + 1);
    for (const auto &e : times) {
        graph[e[0]].push_back({e[1], e[2]});
    }
    const int INF = 1e9;
    std::vector<int> dist(n + 1, INF);
    dist[k] = 0;
    using P = std::pair<int, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    heap.push({0, k});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) {
            continue;
        }
        for (auto [v, w] : graph[u]) {
            int nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                heap.push({nd, v});
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; ++i) {
        if (dist[i] == INF) {
            return -1;
        }
        ans = std::max(ans, dist[i]);
    }
    return ans;
}
```

- **复杂度**：时间 O(E log V)，空间 O(V + E)。
- **易错点**：节点编号是 1~n，数组要开 n+1；堆里可能有过期条目，弹出后要 `if d > dist[u]:
  continue` 跳过；C++ 小顶堆要写 `greater<pair>`，默认是大顶堆。
- **相似题**：1631（把「边权和」换成「边权最大」，见下）；1976（Dijkstra 上数路径）；
  1631、787 等本篇其它最短路题；`docs/08-heap.md` 讲了优先队列本身。

### 1631. 最小体力消耗路径（中等）

**题目**：给一个高度矩阵，从左上走到右下（只能上下左右）。一条路径的体力消耗是路径
上所有相邻格高度差绝对值的**最大值**。求最小的体力消耗。

**思路**：

这里的路径代价不是「边权之和」，而是「路径上最大的一条边」，这类问题叫**瓶颈路**。
只需把 Dijkstra 的松弛公式从加法换成取最大：

    dist[v] = min(dist[v], max(dist[u], |h[v] - h[u]|))

含义是：到 v 的瓶颈，等于「到 u 的瓶颈」和「u→v 这条边的落差」里较大的那个。仍然用
优先队列每次取瓶颈最小的点扩展，正确性来自「max 对非负边权单调」——沿路加边不会让
瓶颈变小，和 Dijkstra 的贪心论证一致。

**代码**（`src/shortest-path/minimum_effort_path.py` / `.cpp`）：

```python
import heapq

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def minimum_effort_path(heights):
    m, n = len(heights), len(heights[0])
    INF = float("inf")
    dist = [[INF] * n for _ in range(m)]
    dist[0][0] = 0
    heap = [(0, 0, 0)]
    while heap:
        effort, r, c = heapq.heappop(heap)
        if r == m - 1 and c == n - 1:
            return effort
        if effort > dist[r][c]:
            continue
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                ne = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if ne < dist[nr][nc]:
                    dist[nr][nc] = ne
                    heapq.heappush(heap, (ne, nr, nc))
    return dist[m - 1][n - 1]
```

```cpp
int minimumEffortPath(const std::vector<std::vector<int>> &heights) {
    int m = heights.size(), n = heights[0].size();
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, INF));
    dist[0][0] = 0;
    using T = std::tuple<int, int, int>;
    std::priority_queue<T, std::vector<T>, std::greater<T>> heap;
    heap.push({0, 0, 0});
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!heap.empty()) {
        auto [effort, r, c] = heap.top();
        heap.pop();
        if (r == m - 1 && c == n - 1) {
            return effort;
        }
        if (effort > dist[r][c]) {
            continue;
        }
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                int ne = std::max(effort, std::abs(heights[nr][nc] - heights[r][c]));
                if (ne < dist[nr][nc]) {
                    dist[nr][nc] = ne;
                    heap.push({ne, nr, nc});
                }
            }
        }
    }
    return dist[m - 1][n - 1];
}
```

- **复杂度**：时间 O(m·n·log(m·n))，空间 O(m·n)。
- **易错点**：松弛里是 `max` 不是 `+`；一开始容易想成「二分最大落差 + BFS」（那也对，
  但复杂度多一个 log）；C++ 的元组小顶堆用 `greater<tuple>` 即可按第一个元素比较。
- **相似题**：778（本篇模式六，同样是「最小化路径上的最大值」，可用二分）；744. 网络
  延迟时间变体；1102. 得分最高的路径（把 max 换成 min 的对称题）。

### 1976. 到达目的地的方案数（中等）

**题目**：n 个城市，道路双向、有耗时。求从 0 到 n-1 在「总耗时最短」的前提下有多少条
不同路径，答案对 1e9+7 取模。

**思路**：

要求「最短路 + 路径条数」，就在 Dijkstra 的同时多维护一个数组 `ways[v]`：0 到 v 的最短
路径条数。扩展边 u→v 时：

- `dist[u]+w < dist[v]`：发现更短的路，`dist[v]` 更新，`ways[v] = ways[u]`；
- `dist[u]+w == dist[v]`：又找到一条等长的路，`ways[v] += ways[u]`。

因为边权非负，所有能作为 v 最短前驱的 u 都会在 v 出堆之前被处理完，所以 v 出堆时
`ways[v]` 已经数齐了。

**代码**（`src/shortest-path/number_of_ways_to_arrive_at_destination.py` / `.cpp`）：

```python
import heapq
from collections import defaultdict

_MOD = 10 ** 9 + 7


def count_paths(n, roads):
    graph = defaultdict(list)
    for u, v, w in roads:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dist = [INF] * n
    ways = [0] * n
    dist[0] = 0
    ways[0] = 1
    heap = [(0, 0)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                ways[v] = ways[u]
                heapq.heappush(heap, (nd, v))
            elif nd == dist[v]:
                ways[v] = (ways[v] + ways[u]) % _MOD
    return ways[n - 1] % _MOD
```

```cpp
const long long MOD = 1000000007LL;

int countPaths(int n, const std::vector<std::vector<int>> &roads) {
    std::vector<std::vector<std::pair<int, int>>> graph(n);
    for (const auto &e : roads) {
        graph[e[0]].push_back({e[1], e[2]});
        graph[e[1]].push_back({e[0], e[2]});
    }
    std::vector<long long> dist(n, LLONG_MAX), ways(n, 0);
    using P = std::pair<long long, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
    dist[0] = 0;
    ways[0] = 1;
    heap.push({0, 0});
    while (!heap.empty()) {
        auto [d, u] = heap.top();
        heap.pop();
        if (d > dist[u]) {
            continue;
        }
        for (auto [v, w] : graph[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                ways[v] = ways[u];
                heap.push({nd, v});
            } else if (nd == dist[v]) {
                ways[v] = (ways[v] + ways[u]) % MOD;
            }
        }
    }
    return static_cast<int>(ways[n - 1] % MOD);
}
```

- **复杂度**：时间 O(E log V)，空间 O(V + E)。
- **易错点**：这是**无向图**，加边要加两次；`ways` 只在「相等」时累加，注意取模；
  起点 `ways[0] = 1`（空路径也算一条）；C++ 距离用 `long long` 防累加溢出。
- **相似题**：`docs/13-dynamic-programming.md` 的「数方案」思想（这里是图上的版本）；
  1786. 从第一个节点出发到最后一个节点的受限路径数（先在 DAG 上 DP）；787（最短路 +
  限制，见下）。

---

## 模式三：边权只有 0/1 —— 0-1 BFS（双端队列）

**适用信号**：每一步的代价非 0 即 1，求最小代价。

**核心动作**：还是 BFS 的队列，但换成**双端队列**——走 0 权边压队首（同层），走 1 权边
压队尾。这样队列里的节点距离最多差 1，出队顺序等价于 Dijkstra，但省掉了堆的 log。

### 1368. 使网格图至少有一条有效路径的最小代价（困难）

**题目**：网格里每格有一个路标方向（1 右、2 左、3 下、4 上）。可以花 1 的代价修改任意
一格的路标。从左上角出发，只能沿所站格的路标走，求让右下角可达的最小代价。

**思路**：

从 (r,c) 走向四个邻居时，若邻居正好是 (r,c) 路标指向的方向，代价 0；否则要改这格的
路标，代价 1。于是得到一张边权只有 0 和 1 的图，求最短路。

既然边权非 0 即 1，用 **0-1 BFS** 最合适：走 0 权边把新点压到**队首**（保持它和当前点
同层），走 1 权边压到**队尾**。这样队列中节点的距离始终非递减，出队就等于取最小，
和 Dijkstra 一样正确。仍用 `nd < dist[nr][nc]` 判断松弛，允许一个点因找到更优解而再次
入队。

**代码**（`src/shortest-path/minimum_cost_to_make_at_least_one_valid_path.py` / `.cpp`）：

```python
from collections import deque

# 下标 0/1/2/3 分别对应路标 1/2/3/4：右、左、下、上
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def min_cost(grid):
    m, n = len(grid), len(grid[0])
    INF = float("inf")
    dist = [[INF] * n for _ in range(m)]
    dist[0][0] = 0
    dq = deque([(0, 0)])
    while dq:
        r, c = dq.popleft()
        for i, (dr, dc) in enumerate(_DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                cost = 0 if grid[r][c] == i + 1 else 1
                nd = dist[r][c] + cost
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    if cost == 0:
                        dq.appendleft((nr, nc))
                    else:
                        dq.append((nr, nc))
    return dist[m - 1][n - 1]
```

```cpp
int minCost(const std::vector<std::vector<int>> &grid) {
    int m = grid.size(), n = grid[0].size();
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, INF));
    dist[0][0] = 0;
    std::deque<std::pair<int, int>> dq;
    dq.push_back({0, 0});
    // 下标 0/1/2/3 分别对应路标 1/2/3/4：右、左、下、上
    const int dr[4] = {0, 0, 1, -1};
    const int dc[4] = {1, -1, 0, 0};
    while (!dq.empty()) {
        auto [r, c] = dq.front();
        dq.pop_front();
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                int cost = (grid[r][c] == i + 1) ? 0 : 1;
                int nd = dist[r][c] + cost;
                if (nd < dist[nr][nc]) {
                    dist[nr][nc] = nd;
                    if (cost == 0) {
                        dq.push_front({nr, nc});
                    } else {
                        dq.push_back({nr, nc});
                    }
                }
            }
        }
    }
    return dist[m - 1][n - 1];
}
```

- **复杂度**：时间 O(m·n)（每点至多入队两次），空间 O(m·n)。
- **易错点**：0 权边必须压**队首**、1 权边压队尾，写反就退化成普通 BFS；方向数组的下标
  要和路标数字 1/2/3/4 对应上；被松弛的点可能已经在队列里，再压一次没问题（旧条目出队
  时因 `nd` 不再是当前 `dist` 会被自然忽略）。
- **相似题**：2290. 到达角落需要移除障碍物的最小数目（0-1 BFS 名题）；1293. 网格中的
  最短路径（BFS + 消除障碍，属于分层 BFS）。

---

## 模式四：边数/时间有限制 —— 按轮松弛（Bellman-Ford 思想）

**适用信号**：题目限制「最多 k 步 / k 次中转 / 规定时间」，要求在此限制下最优。

**核心动作**：把「限制」当成状态的一个维度，做若干轮松弛；每轮基于**上一轮的快照**更新，
这样第 i 轮的结果恰好表示「用不超过 i 步/时间」的最优值。这种「不许在同一轮里连续
延长」的写法，正是 Bellman-Ford 的精髓。

### 787. K 站中转内最便宜的航班（中等）

**题目**：n 个城市，有向航线 `[from, to, price]`。求从 src 到 dst 在「最多 k 个中转站」
下的最便宜价格，不可达返回 -1。

**思路**：

「最多 k 个中转」等价于「路径至多 k+1 条边」。做 k+1 轮松弛，第 i 轮结束时 `dist` 表示
「用不超过 i 条边能到达的最低价格」。

**关键**：每轮必须先 `prev = dist[:]`，松弛时读 `prev`、写 `dist`。如果直接在已更新的
`dist` 上继续松弛，一条路径会在一轮内被反复延长，等于用了任意多条边，k 的限制就形同
虚设——这是本题最容易写错的地方。

**代码**（`src/shortest-path/cheapest_flights_within_k_stops.py` / `.cpp`）：

```python
def find_cheapest_price(n, flights, src, dst, k):
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        prev = dist[:]
        for u, v, w in flights:
            if prev[u] != INF and prev[u] + w < dist[v]:
                dist[v] = prev[u] + w
    return -1 if dist[dst] == INF else dist[dst]
```

```cpp
int findCheapestPrice(int n, const std::vector<std::vector<int>> &flights, int src,
                      int dst, int k) {
    const long long INF = LLONG_MAX;
    std::vector<long long> dist(n, INF);
    dist[src] = 0;
    for (int round = 0; round <= k; ++round) {
        std::vector<long long> prev = dist;
        for (const auto &e : flights) {
            int u = e[0], v = e[1], w = e[2];
            if (prev[u] != INF && prev[u] + w < dist[v]) {
                dist[v] = prev[u] + w;
            }
        }
    }
    return dist[dst] == INF ? -1 : static_cast<int>(dist[dst]);
}
```

- **复杂度**：时间 O(k·E)，空间 O(V)。
- **易错点**：一定要用快照 `prev`，不能原地更新；松弛次数是 k+1（中转 k 个 ⟺ 边数 ≤ k+1）；
  `prev[u]` 为无穷时要跳过，避免无穷加边溢出。
- **相似题**：1928（把「边数」换成「时间」，见下）；`docs/13-dynamic-programming.md`
  的「分层/状态扩展」思想；743（若给每条边加一层「已用中转数」也能做，但复杂度更高）。

### 1928. 规定时间内到达终点的最小花费（困难）

**题目**：n 个城市，双向边带耗时。`passingFees[i]` 是经过城市 i 要交的过路费（起点终点
也要交）。从城市 0 出发，在总耗时不超过 maxTime 的前提下到城市 n-1，求最小总花费。

**思路**：

只记「到城市 v 的最小花费」不够用——花费小的路可能特别慢，后面就超时了。所以把**时间**
也放进状态：`dp[t][v] = 不超过耗时 t 到达 v 的最小花费`。转移就是走一条边：

    dp[t+w][v] = min(dp[t+w][v], dp[t][u] + passingFees[v])

初始 `dp[0][0] = passingFees[0]`（起点也要交费）。按 t 从小到大遍历即可，因为转移总
把时间推大，天然满足拓扑序，不需要堆。这其实就是在一张「(城市, 时间)」的分层图上递推，
和模式四「给限制加一个维度」是同一个思路。

**代码**（`src/shortest-path/minimum_cost_to_reach_destination_in_time.py` / `.cpp`）：

```python
def min_cost(max_time, edges, passing_fees):
    n = len(passing_fees)
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    INF = float("inf")
    dp = [[INF] * n for _ in range(max_time + 1)]
    dp[0][0] = passing_fees[0]
    for t in range(max_time + 1):
        for u in range(n):
            if dp[t][u] == INF:
                continue
            for v, w in graph[u]:
                nt = t + w
                if nt <= max_time and dp[t][u] + passing_fees[v] < dp[nt][v]:
                    dp[nt][v] = dp[t][u] + passing_fees[v]

    ans = min(dp[t][n - 1] for t in range(max_time + 1))
    return -1 if ans == INF else ans
```

```cpp
int minCost(int maxTime, const std::vector<std::vector<int>> &edges,
            const std::vector<int> &passingFees) {
    int n = passingFees.size();
    std::vector<std::vector<std::pair<int, int>>> graph(n);
    for (const auto &e : edges) {
        graph[e[0]].push_back({e[1], e[2]});
        graph[e[1]].push_back({e[0], e[2]});
    }
    const int INF = 1e9;
    std::vector<std::vector<int>> dp(maxTime + 1, std::vector<int>(n, INF));
    dp[0][0] = passingFees[0];
    for (int t = 0; t <= maxTime; ++t) {
        for (int u = 0; u < n; ++u) {
            if (dp[t][u] == INF) {
                continue;
            }
            for (auto [v, w] : graph[u]) {
                int nt = t + w;
                if (nt <= maxTime && dp[t][u] + passingFees[v] < dp[nt][v]) {
                    dp[nt][v] = dp[t][u] + passingFees[v];
                }
            }
        }
    }
    int ans = INF;
    for (int t = 0; t <= maxTime; ++t) {
        if (dp[t][n - 1] < ans) {
            ans = dp[t][n - 1];
        }
    }
    return ans == INF ? -1 : ans;
}
```

- **复杂度**：时间 O(maxTime·E)，空间 O(maxTime·n)。
- **易错点**：起点也要收过路费（`dp[0][0] = passingFees[0]`）；终点也是「经过」要收费；
  按时间从小到大递推，别写成 Dijkstra 的堆（虽然也能做）；最后答案是所有时间上的最小值，
  不是 `dp[maxTime][n-1]`。
- **相似题**：787（本条的模式四「姊妹题」）；`docs/13-dynamic-programming.md` 的
  「状态里放第二个维度」；`docs/04-prefix-sum.md` 的差分无关，但同样体现「换维思考」。

---

## 模式五：任意两点间最短路 —— Floyd-Warshall

**适用信号**：要对**每一对**城市都求最短路（或「每个点的邻居数」这类全源统计）。

**核心动作**：用「允许经过哪些中转点」逐层放松，三重循环，中转点 k 放在最外层：

    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

### 1334. 阈值距离内邻居最少的城市（中等）

**题目**：n 个城市，双向边带距离。给定阈值 `distanceThreshold`，若 i 到 j 的最短距离
不超过阈值就称 j 是 i 的邻居。求邻居数最少的城市，并列时取编号最大的；要求至少有
一个邻居。

**思路**：

要对每个城市都算出到其他所有城市的最短路，这正是**全源最短路**，Floyd-Warshall 最直接。
外层枚举中转点 k，表示「只允许经过编号 ≤ k 的点」时的最短路；k 扫完就是全局最短。

**为什么 k 必须在最外层**：Floyd 的正确性依赖「第 k 轮只放开 k 这一个新中转点」的递推。
如果 k 写在最内层，相当于在计算 `dist[i][j]` 时随意借用还没算好的 `dist[i][k]`，会过早
固化错误答案。然后统计每个点在阈值内的邻居数，用 `<=` 比较即可在并列时自动选出编号
更大的城市。

**代码**（`src/shortest-path/find_the_city_with_the_smallest_number_of_neighbors.py` / `.cpp`）：

```python
def find_the_city(n, edges, distance_threshold):
    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
        if w < dist[v][u]:
            dist[v][u] = w

    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    best_city = -1
    best_count = n + 1
    for i in range(n):
        cnt = sum(
            1 for j in range(n) if i != j and dist[i][j] <= distance_threshold
        )
        if cnt <= best_count:
            best_count = cnt
            best_city = i
    return best_city
```

```cpp
int findTheCity(int n, const std::vector<std::vector<int>> &edges,
                int distanceThreshold) {
    const int INF = 1e9;
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, INF));
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
    }
    for (const auto &e : edges) {
        int u = e[0], v = e[1], w = e[2];
        dist[u][v] = std::min(dist[u][v], w);
        dist[v][u] = std::min(dist[v][u], w);
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (dist[i][k] == INF) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    int bestCity = -1, bestCount = n + 1;
    for (int i = 0; i < n; ++i) {
        int cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j && dist[i][j] <= distanceThreshold) {
                ++cnt;
            }
        }
        if (cnt <= bestCount) {
            bestCount = cnt;
            bestCity = i;
        }
    }
    return bestCity;
}
```

- **复杂度**：时间 O(n³)，空间 O(n²)。
- **易错点**：中转点 k 必须最外层；`dist[i][i] = 0`；重边要取较小值；统计邻居时排除
  `i == j`；并列取编号大的城市，所以用 `<=` 而不是 `<`。
- **相似题**：743（单源版本，用 Dijkstra 即可，别用 O(n³) 的 Floyd）；399. 除法求值
  （把比值乘法转成对数加法后同样能跑 Floyd 式传递闭包）；`docs/10-graph.md`。

---

## 模式六：把最优化转成判定 —— 二分答案 + BFS

**适用信号**：求「最早的时刻 / 最小的上限」这类答案，而「给定一个值能否做到」很容易
判断，且判断结果随值单调。

**核心动作**：二分答案值，每次用一个更简单的子程序（这里是 BFS 连通性）判定可行性。
二分能把「求最优」变成「判可行」的若干次调用。

### 778. 水位上升的泳池中游泳（困难）

**题目**：n×n 方格，`grid[i][j]` 是平台高度（0~n²-1 的一个排列）。时刻 t 水位为 t，
只有高度 ≤ t 的平台能游。求从左上到右下最早能到达的时刻。

**思路**：

「最早时刻」不好直接求，但「在时刻 t 能否到达终点」很好判定：只保留高度 ≤ t 的格子，
做一次 BFS 看起终点是否连通。而且这个判定关于 t **单调**——t 越大可用格子越多，能到达
只会从「不能」变成「能」，不会反向。于是二分最小的可行 t。

答案下界是 `max(grid[0][0], grid[n-1][n-1])`（起点终点本身得能被淹），上界是 `n²-1`
（最大高度）。二分时若 `mid` 可行就收缩上界，否则抬高下界。

**为什么二分适用**：单调性是二分的命根子。「不可行 → 可行」只发生一次，数轴上表现为
一段「否」接一段「是」，二分每次比较中点就能砍掉一半区间。

**代码**（`src/shortest-path/swim_in_rising_water.py` / `.cpp`）：

```python
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def swim_in_water(grid):
    n = len(grid)

    def can_reach(t):
        if grid[0][0] > t or grid[n - 1][n - 1] > t:
            return False
        seen = [[False] * n for _ in range(n)]
        seen[0][0] = True
        queue = deque([(0, 0)])
        while queue:
            r, c = queue.popleft()
            if r == n - 1 and c == n - 1:
                return True
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and not seen[nr][nc]
                    and grid[nr][nc] <= t
                ):
                    seen[nr][nc] = True
                    queue.append((nr, nc))
        return False

    lo = max(grid[0][0], grid[n - 1][n - 1])
    hi = n * n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if can_reach(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

```cpp
bool canReach(const std::vector<std::vector<int>> &grid, int t) {
    int n = grid.size();
    if (grid[0][0] > t || grid[n - 1][n - 1] > t) {
        return false;
    }
    std::vector<std::vector<bool>> seen(n, std::vector<bool>(n, false));
    seen[0][0] = true;
    std::queue<std::pair<int, int>> q;
    q.push({0, 0});
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        if (r == n - 1 && c == n - 1) {
            return true;
        }
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < n && nc >= 0 && nc < n && !seen[nr][nc] &&
                grid[nr][nc] <= t) {
                seen[nr][nc] = true;
                q.push({nr, nc});
            }
        }
    }
    return false;
}

int swimInWater(const std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    int lo = std::max(grid[0][0], grid[n - 1][n - 1]);
    int hi = n * n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (canReach(grid, mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
```

- **复杂度**：时间 O(n²·log n)，空间 O(n²)。
- **易错点**：二分下界必须是起终点高度的较大者；`while lo < hi` + `hi = mid` 是「找最小
  可行值」的模板，别写成 `hi = mid - 1`；判定函数要判起终点是否被淹。本题也能用
  Dijkstra（瓶颈路，见本篇 1631），复杂度相当。
- **相似题**：`docs/05-binary-search.md` 的「二分答案」模板（875、1011）；1631（同一类
  瓶颈路，用 Dijkstra 直接求）；1102. 得分最高的路径。

---

## 规律总结

1. **先判边权，再选算法**：边权全 1 → BFS；非负 → Dijkstra；只有 0/1 → 0-1 BFS；
   有「步数/时间」限制 → 按轮松弛（Bellman-Ford）；要全源 → Floyd-Warshall；
   「能不能」易判且单调 → 二分答案。选型的依据是**边的性质**，不是题目表象。

2. **BFS 是最短路的特例**：普通 BFS 本质上就是「边权全为 1 的 Dijkstra」。把 FIFO 队列
   换成小顶堆、把「层数」换成「路径和」，就得到 Dijkstra。理解这一点，两种算法都不必死背。

3. **0-1 BFS 是「不用堆的 Dijkstra」**：边权非 0 即 1 时，队列里节点距离最多差 1，用
   双端队列把 0 权边压队首、1 权边压队尾，就能维持出队有序，省掉 log。

4. **限制条件要变成状态的一个维度**：787 的「中转次数」是层数，1928 的「时间」是第二个
   下标。凡是「在某个约束下最优」，都先想「把约束做成状态」。

5. **按轮松弛要用快照**：Bellman-Ford / 分层 DP 里，每轮必须基于上一轮的值更新，否则会
   在一轮内连续延长路径，绕过「最多 k 步」的限制。这是最容易踩的坑。

6. **全源用 Floyd，但注意适用范围**：Floyd 是 O(n³)，只适合 n 较小；单源最短路不要用
   Floyd，Dijkstra（非负）或 Bellman-Ford（带负权/限步）更快。Floyd 的三重循环里中转点
   k 必须最外层。

7. **瓶颈路（最小化路径最大值）可改 Dijkstra，也可二分**：把松弛的加法换成取 max 就是
   Dijkstra 版；把「最早时刻」当成可判定问题就得到二分 + BFS 版。两种方法常互为替代，
   选更顺手的。

8. **Dijkstra 的松弛可以是任意单调运算**：加法、取 max、取 min 都能套，只要「沿路径继续
   走不会让结果变好」的单调性成立。这说明最短路算法本身比「距离」这个词更通用——它其实
   是在一张图上求「可按单调量比较的最优路径」。

9. **网格是隐式图**：最短路的一大半题在网格上。把每个格子当节点、相邻当边，网格题就变成
   图题。多源、八方向、带障碍都只是加源/改方向/加判定，主干不变（见 `docs/10-graph.md`）。

10. **路径计数 = 最短路 + 计数**：1976 在 Dijkstra 上多维护一个 `ways` 数组，发现更短
    就重置、发现等长就累加。这个模式和 DP 里的「数方案」一脉相承。
