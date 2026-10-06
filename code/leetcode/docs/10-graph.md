# 图与搜索：网格遍历、拓扑排序与并查集

「图」听起来抽象，但面试里最常见的图根本不用建邻接表——**网格（二维数组）本身就是一张图**。
每个格子是一个节点，上下左右相邻就有一条边。岛屿、迷宫、染色、连通区域这类题，
都是在问「这块网格里有几个连通块」「某一块有多大」，用一次 **DFS 或 BFS 把相连的格子走一遍**
就能解决。

本篇从这套最基础的「网格搜索」出发，再补上几件**图论的常用工具**：多源 BFS 求最短距离、
拓扑排序处理依赖、并查集维护连通块、以及「从边界反向标记」的巧思。
它们看上去各成一套，其实都由同一个问题驱动——**节点之间怎么连通、按什么顺序走**。

先记住两个恒定的小零件：

- **方向**：四方向偏移 `(-1,0) (1,0) (0,-1) (0,1)`，或者直接写四次递归；
- **越界判断**：`r < 0 || r >= rows || c < 0 || c >= cols` 必须写在最前面，
  否则会访问到非法内存（C++ 直接崩溃）。

本篇题目（由易到难）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：网格 DFS 标记连通块 | 200. 岛屿数量 | 中等 |
| 模式一：网格 DFS 标记连通块 | 733. 图像渲染 | 简单 |
| 模式一：网格 DFS 标记连通块 | 695. 岛屿的最大面积 | 中等 |
| 模式二：图的克隆与遍历 | 133. 克隆图 | 中等 |
| 模式三：多源 BFS 求最短距离 | 994. 腐烂的橘子 | 中等 |
| 模式三：多源 BFS 求最短距离 | 542. 01 矩阵 | 中等 |
| 模式四：拓扑排序 | 207. 课程表 | 中等 |
| 模式四：拓扑排序 | 210. 课程表 II | 中等 |
| 模式五：并查集 | 547. 省份数量 | 中等 |
| 模式五：并查集 | 684. 冗余连接 | 中等 |
| 模式六：从边界反向标记 | 130. 被围绕的区域 | 中等 |

---

## 模式一：网格 DFS，把整个连通块「淹没」

**适用信号**：题目给出一个二维网格，问「有多少个」「某一块多大」「把某块改成什么」，
且相连的定义是「上下左右」。关键词常带「岛屿 / 连通 / 区域 / 渲染 / 相邻」。

**核心动作**：写一个 `dfs(r, c)`，表示「从格子 (r, c) 出发，把与它相连的同类格子全部处理掉」。
进入函数先做三件事——越界就返回、不属于目标类型就返回、否则标记当前格「已处理」——
然后对上下左右四个方向各递归一次。

**为什么「标记」放在递归之前**：标记既是「记录已访问」，也是防止两个相邻格子互相递归
造成无限循环的唯一屏障。不标记就会在 A↔B 之间来回横跳，栈溢出。

**为什么可以直接改原网格**：访问过的格子对后续没有价值，就地涂改既完成了标记，
又省下一个 `visited` 表。如果不允许修改输入，再额外开一个布尔数组即可。

### 200. 岛屿数量（中等）

**题目**：给你一个由字符 `'1'`（陆地）和 `'0'`（水）组成的二维网格 `grid`，
请计算网格中岛屿的数量。岛屿被水包围，由水平或垂直方向相邻的陆地连接而成，
网格的四条边均被水包围。

**思路**：
把每个陆地格子当成图的节点，相邻陆地之间连边，那么「岛屿数量」**就等于「连通块数量」**。
于是做法很直接：

1. 从上到下、从左到右扫描整个网格；
2. 遇到一个 `'1'`，说明碰上了一座还没数过的岛——计数加一；
3. 立刻从这个格子出发 DFS，把整座岛的 `'1'` 全部改成 `'0'`（淹没）；
4. 继续扫描；因为同一座岛已被淹没，不会再被数第二次。

每个格子只会被 DFS 访问一次，所以时间是 O(m·n)。就地涂改省掉了 `visited` 表，
但递归栈最坏（整张图全是陆地）会达到 O(m·n)——这是 DFS 版本唯一需要注意的代价，
担心栈溢出时可以改用显式栈或队列的迭代写法。

**代码**（完整可运行版见 `src/graph/number_of_islands.py` / `.cpp`）：

```python
def num_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                count += 1
                dfs(r, c)
    return count
```

```cpp
void dfs(std::vector<std::vector<char>> &grid, int r, int c) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') return;
    grid[r][c] = '0';
    dfs(grid, r + 1, c);
    dfs(grid, r - 1, c);
    dfs(grid, r, c + 1);
    dfs(grid, r, c - 1);
}

int numIslands(std::vector<std::vector<char>> &grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int rows = grid.size();
    int cols = grid[0].size();
    int count = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == '1') {
                ++count;
                dfs(grid, r, c);
            }
        }
    }
    return count;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（递归栈最坏情形）。
- **易错点**：边界判断顺序——先判越界再判 `grid[r][c]`，否则数组下标已经越界；
  空网格 `[]` 或 `[[]]` 要先挡掉，不然 `grid[0][0]` 会抛异常；
  每发现新岛时先计数再 DFS，别写反；用 `"1"`（字符）不要写成 `1`（数字），
  Python 不会自动转换。
- **相似题**：695. 岛屿的最大面积（同一框架，DFS 返回面积）、733. 图像渲染（同一框架，
  把连通块染色）、130. 被围绕的区域（先用 DFS 标记边缘岛，再把内部岛改写，见后续图专题）。

### 733. 图像渲染（简单）

**题目**：有一幅以二维整数数组表示的图画 `image`，`image[i][j]` 表示该位置像素的颜色。
给你三个整数 `sr`、`sc`、`color`，从坐标 `(sr, sc)` 开始渲染：把所有与起始像素颜色相同、
四方向相连的像素都染成 `color`，返回渲染后的图像。

**思路**：
这题本质还是「找连通块」，只是终点从「计数」换成「染色」。做法与 200 如出一辙：
从 `(sr, sc)` 出发 DFS，凡是颜色等于**起始颜色**的格子，一律改成新颜色。

有一个必须提前处理的陷阱：**如果起始颜色恰好等于新颜色，扩散条件永远成立、
颜色又始终不变，就会无限递归**。所以在进入 DFS 前先判断，相等就直接返回原图。

另一点：DFS 里判断的条件是「颜色仍等于起始颜色」。被染过的格子颜色已经变了，
自然不再满足条件，这就顺带完成了「已访问」标记，不需要额外的 visited 表。

**代码**（`src/graph/flood_fill.py` / `.cpp`）：

```python
def flood_fill(image, sr, sc, color):
    rows, cols = len(image), len(image[0])
    start = image[sr][sc]
    if start == color:
        return image

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or image[r][c] != start:
            return
        image[r][c] = color
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    dfs(sr, sc)
    return image
```

```cpp
void dfs(std::vector<std::vector<int>> &image, int r, int c, int start, int color) {
    int rows = image.size();
    int cols = image[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || image[r][c] != start) return;
    image[r][c] = color;
    dfs(image, r + 1, c, start, color);
    dfs(image, r - 1, c, start, color);
    dfs(image, r, c + 1, start, color);
    dfs(image, r, c - 1, start, color);
}

std::vector<std::vector<int>> floodFill(std::vector<std::vector<int>> image,
                                        int sr, int sc, int color) {
    int start = image[sr][sc];
    if (start == color) return image;
    dfs(image, sr, sc, start, color);
    return image;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)。
- **易错点**：忘记 `start == color` 的提前返回，导致无限递归——这是本题第一大坑；
  比较对象是「起始颜色」而不是「上一次的颜色」，两者在只有两种颜色时容易混淆；
  C++ 版把 `image` 按值传参以返回新图（与 Python 就地修改后返回保持一致的对外语义），
  若改成引用传参，返回类型与用法都要相应调整。
- **相似题**：200. 岛屿数量、695. 岛屿的最大面积（同模板，只是最终产物不同）；
  1254. 统计封闭岛屿的数目（在网格 DFS 上多加一步边缘判断，见后续图专题）。

### 695. 岛屿的最大面积（中等）

**题目**：给你一个大小为 m x n 的二进制矩阵 `grid`，`1` 表示陆地、`0` 表示水。
岛屿是四方向相连的 `1` 组成的连通块。返回网格中岛屿的最大面积；没有岛屿则返回 0。

**思路**：
扫描框架和 200 完全一样，仍是「遇到未访问的 `1` 就发现一座新岛」。
区别在于要问「这座岛有多大」，于是让 **DFS 直接把面积作为返回值**：

- 踩到越界或水，返回 0（加法的单位元）；
- 站在陆地上，先把自己标记为 `0`，再返回 `1 + 四个方向 DFS 的和`。

这样整座岛的面积就由递归一层层汇总到入口，外层用 `max` 取所有岛的最大值。
为什么用返回值而不是全局变量：每个格子的贡献都恰好是「自己一份」，递归相加天然把
连通块的大小算清，不依赖访问顺序，也不怕多个连通块互相干扰——这正是「后序递归返回值」
思路在网格上的翻版。

**代码**（`src/graph/max_area_of_island.py` / `.cpp`）：

```python
def max_area_of_island(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    best = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                best = max(best, dfs(r, c))
    return best
```

```cpp
int dfs(std::vector<std::vector<int>> &grid, int r, int c) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != 1) return 0;
    grid[r][c] = 0;
    return 1 + dfs(grid, r + 1, c) + dfs(grid, r - 1, c) + dfs(grid, r, c + 1) +
           dfs(grid, r, c - 1);
}

int maxAreaOfIsland(std::vector<std::vector<int>> grid) {
    if (grid.empty() || grid[0].empty()) return 0;
    int rows = grid.size();
    int cols = grid[0].size();
    int best = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 1) {
                best = std::max(best, dfs(grid, r, c));
            }
        }
    }
    return best;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)。
- **易错点**：`best` 初值取 0 而不是负数，这样全水网格能正确返回 0；
  是 `1 + 四方向` 而不是 `四方向`，漏掉自己会少算一个格；
  越界/为水时返回 0，不能返回 1；Python 判断用整数 `1`（本题是整型矩阵），
  别和 200 的字符 `'1'` 搞混。
- **相似题**：200. 岛屿数量（把「面积」换成「个数」）、733. 图像渲染（把「面积」换成「染色」）；
  1034. 边界着色、1020. 飞地的数量（都是网格连通块的变体）。

---

## 模式二：图的克隆与遍历

**适用信号**：题目给的是**显式的图结构**（节点用 `neighbors` 指针或邻接表表示），
而不是网格；常见问法是「深拷贝」「复制」「遍历所有节点」。这时要先在脑子里把
「原节点」和「克隆节点」区分开，再用一张哈希表把两者对上号。

### 133. 克隆图（中等）

**题目**：给你无向连通图中一个节点的引用 `node`，返回该图的深拷贝（克隆）。
每个节点的值等于它的编号，节点用 `neighbors` 列表保存所有邻居的引用。

**思路**：
深拷贝最大的风险是**同一个原节点被克隆多次**——尤其在环里，A 的邻居是 B、B 的邻居又是 A，
没有记录就会无限复制下去，克隆图也会和原图不同构。所以核心是一张哈希表
`clones`，键是原节点、值是它的克隆节点，保证「每个原节点只克隆一次」。

从 `node` 出发做 DFS：

1. 若当前节点已经在 `clones` 里，直接返回它的克隆（这就是去重）；
2. 否则新建一个**只带值、还没有邻居**的克隆，先登记进 `clones`；
3. 再逐个克隆它的邻居，把克隆出的邻居接进克隆节点的 `neighbors`。

**为什么先登记、再递归邻居**：这是本题的关键顺序。递归邻居时可能绕回当前节点
（自环或环），此时能从 `clones` 里取到刚登记的克隆，从而正确建出自环/环的结构；
如果把登记放到递归之后，就会在环上无限递归、栈溢出。

**为什么哈希表用「节点对象」作键**：题目里节点的值恰好等于编号且互不相同，
用值作键也行；但真正稳妥的做法是用对象引用作键——因为深拷贝要复制的正是
「谁和谁相连」这层引用关系，按对象身份对齐才能保证同构。

**代码**（完整可运行版见 `src/graph/clone_graph.py` / `.cpp`）：

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


def clone_graph(node):
    if node is None:
        return None

    clones = {}

    def dfs(cur):
        if cur in clones:
            return clones[cur]
        copy = Node(cur.val)
        clones[cur] = copy
        for nb in cur.neighbors:
            copy.neighbors.append(dfs(nb))
        return copy

    return dfs(node)
```

```cpp
class Node {
public:
    int val;
    std::vector<Node *> neighbors;
    Node() : val(0) {}
    explicit Node(int _val) : val(_val) {}
    Node(int _val, std::vector<Node *> _neighbors)
        : val(_val), neighbors(std::move(_neighbors)) {}
};

Node *dfs(Node *cur, std::map<Node *, Node *> &clones) {
    auto it = clones.find(cur);
    if (it != clones.end()) return it->second;
    Node *copy = new Node(cur->val);
    clones[cur] = copy;
    for (Node *nb : cur->neighbors) {
        copy->neighbors.push_back(dfs(nb, clones));
    }
    return copy;
}

Node *cloneGraph(Node *node) {
    if (node == nullptr) return nullptr;
    std::map<Node *, Node *> clones;
    return dfs(node, clones);
}
```

- **复杂度**：时间 O(V + E)，空间 O(V)。
- **易错点**：忘记处理空图（`node` 为 `null`）会直接解引用崩溃；
  先递归邻居再登记会因环无限递归——**登记必须在前**；
  要用原节点的值建克隆、却忘了把克隆登记进表，下一轮又新建一个，克隆图被复制多份；
  C++ 返回的是 `new` 出来的节点，由调用方负责释放。
- **相似题**：133 是「图的遍历 + 去重」的原型；把「去重表」换成「已访问集合」，
  就是 200. 岛屿数量、547. 省份数量里数连通块的套路。

---

## 模式三：多源 BFS，一次扩散求最短距离

**适用信号**：题目问「最少要多少步 / 多少分钟」「每个点到最近某个东西的距离」，
且扩散是**同时**从多个起点发生的（多个腐烂橘子、多个 0）。关键词常带
「最短 / 最小 / 最近 / 同时」。

**为什么用 BFS 而不是 DFS**：BFS 天然按「距离层」推进——第 k 层就是距离为 k 的所有格子。
所以第一次访问到某个格子的层数，就是它的最短距离。DFS 会一头扎到底，
无法保证先到达的是最近的。

**什么是「多源」**：把**所有**起点在第 0 层一次性放进队列。这等价于在图上虚拟一个源点，
向每个真实起点连一条边，于是「多源最短路」就化成了普通的单源 BFS。

### 994. 腐烂的橘子（中等）

**题目**：`m x n` 网格中，`0` 表示空格、`1` 表示新鲜橘子、`2` 表示腐烂橘子。
每分钟，腐烂橘子会把它上下左右相邻的新鲜橘子变腐烂。返回直到没有新鲜橘子为止
所需的**最小分钟数**；如果不可能全部腐烂，返回 `-1`。

**思路**：
所有腐烂橘子在同一分钟一起向外扩散，正是典型的多源 BFS。

1. 扫描网格：腐烂橘子全部入队，同时数出新鲜橘子数量 `fresh`；
2. 每一轮**处理当前队列里的全部节点**——它们代表同一分钟被感染的橘子；
   每取出一个，就看四个邻居，若是新鲜橘子就变腐烂、`fresh` 减一、入队；
3. 一轮处理完，分钟数加一；
4. 循环结束时若 `fresh == 0`，返回分钟数，否则返回 `-1`。

**为什么要按「当前队列长度」分层**：队里同一时刻的节点都是同一分钟的「感染波前」。
用 `for _ in range(len(queue))` 把这一层一次性处理干净，分钟数才计得准；
若一次只弹一个就加时间，会把同一分钟的感染拆成好几分钟。

**为什么循环条件带 `fresh > 0`**：一旦没有新鲜橘子，后面的扩展毫无意义；
不加这个条件，最后一层结束时 `minutes` 还会凭空多算一分钟。

**代码**（`src/graph/rotting_oranges.py` / `.cpp`）：

```python
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def oranges_rotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    minutes = 0
    while queue and fresh > 0:
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc))
        minutes += 1

    return minutes if fresh == 0 else -1
```

```cpp
int orangesRotting(std::vector<std::vector<int>> grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    std::queue<std::pair<int, int>> q;
    int fresh = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (grid[r][c] == 2) {
                q.push({r, c});
            } else if (grid[r][c] == 1) {
                ++fresh;
            }
        }
    }

    int minutes = 0;
    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty() && fresh > 0) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            auto [r, c] = q.front();
            q.pop();
            for (int d = 0; d < 4; ++d) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                    grid[nr][nc] == 1) {
                    grid[nr][nc] = 2;
                    --fresh;
                    q.push({nr, nc});
                }
            }
        }
        ++minutes;
    }
    return fresh == 0 ? minutes : -1;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)。
- **易错点**：忘记统计 `fresh`，导致无法区分「本来就烂光」和「有橘子永远烂不掉」；
  循环里用 `while queue` 而不是 `while queue and fresh > 0`，结果多算一分钟；
  没按层处理（`len(queue)`）而是每弹一个就 `minutes += 1`，时间被算大；
  全无新鲜橘子时正确结果是 0，别被循环里多加的 1 带偏。
- **相似题**：542. 01 矩阵（同样是多源 BFS，求的是每个点到最近 0 的距离）；
  994 与 200/695 恰好互补——前者求「最短时间」用 BFS，后者求「有几个连通块」用 DFS。

### 542. 01 矩阵（中等）

**题目**：给定 `0/1` 矩阵 `mat`，输出同样大小的矩阵，每个位置是原矩阵对应位置
到最近的 `0` 的距离；相邻格距离为 1。

**思路**：
朴素做法是对每个 `1` 单独 BFS 找最近的 `0`，但每个都要重搜，代价高。反向想：
**所有 `0` 都是距离为 0 的源**，让它们同时向外扩散一层，碰到的 `1` 距离就是 1，
再向外一层是 2……一次多源 BFS 就能填好整张表。

1. 开 `dist` 矩阵，初值全为 `-1`（表示距离尚未确定）；所有 `0` 的位置填 `0` 并入队；
2. 逐个出队，看四个邻居：若邻居的 `dist` 还是 `-1`，它的距离就是当前格距离加一，入队；
3. 队列跑空，所有格子的最短距离都确定了。

**为什么 `dist == -1` 就是「没访问过」**：`-1` 兼作 visited 标记，一举两得。
一个格子只会被最近的那个 `0` 第一次填上，之后再被访问不会更短，
所以「首次访问即最优」——这正是 BFS 按距离分层的性质。

**代码**（`src/graph/update_matrix.py` / `.cpp`）：

```python
from collections import deque

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def update_matrix(mat):
    rows, cols = len(mat), len(mat[0])
    dist = [[-1] * cols for _ in range(rows)]
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dist[r][c] = 0
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist
```

```cpp
std::vector<std::vector<int>> updateMatrix(const std::vector<std::vector<int>> &mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    std::vector<std::vector<int>> dist(rows, std::vector<int>(cols, -1));
    std::queue<std::pair<int, int>> q;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (mat[r][c] == 0) {
                dist[r][c] = 0;
                q.push({r, c});
            }
        }
    }

    const int dr[4] = {1, -1, 0, 0};
    const int dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                dist[nr][nc] == -1) {
                dist[nr][nc] = dist[r][c] + 1;
                q.push({nr, nc});
            }
        }
    }
    return dist;
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)。
- **易错点**：用「对每个 1 单独 BFS」也能过，但复杂度和代码都更重，多源才是正解；
  忘记把队列按层处理不影响本题结果（这里不需要分钟数），但混用会写乱；
  `dist` 初值别用 0，否则无法区分「距离为 0 的 0」和「尚未访问」；
  本题保证至少有一个 `0`；若真出现全 `1` 网格，会留下 `-1`。
- **相似题**：994. 腐烂的橘子（同一模板求时间）；542 也可以从四个方向做两遍
  动态规划（正向一遍、反向一遍取 min），思路等价但代码更长，不如多源 BFS 直观。

---

## 模式四：拓扑排序，给有依赖的节点排个序

**适用信号**：题目里有一组「先后依赖」关系（先修课、编译顺序、任务调度），
问「能不能全部完成」或「给出一个合法顺序」。把依赖画成**有向图**，问的就是
「有没有环」以及「没有环时的一个线性序」。

**Kahn 算法（BFS 版拓扑排序）三步**：

1. 统计每个点的**入度**（有多少条边指向它），把入度为 0 的点全部入队；
2. 反复取出队首，加入结果序列，并把它指向的所有点的入度减一；谁减到 0 就入队；
3. 若结果序列的长度等于节点总数，说明无环；否则剩下的点都在环上。

**为什么入度为 0 就代表「可以做了」**：指向它的所有依赖都已经完成，没有未满足的前置；
**为什么有环就排不完**：环上的每个点都在等环上的另一个点，入度永远降不到 0，
永远无法入队，所以最终序列一定短于总数。

### 207. 课程表（中等）

**题目**：需要修 `numCourses` 门课（编号 `0 ~ numCourses-1`）。
`prerequisites[i] = [a, b]` 表示修 `a` 之前必须先修 `b`。判断能否修完所有课程。

**思路**：
把课程看作节点，先修关系 `b -> a` 看作有向边，「能否修完」等价于「图中无环」。
直接套 Kahn 算法：入度为 0 的课先修，修一门就把它后续课程的入度减一，
最后看修掉的课程数是否等于总数。

**为什么用入度而不是出度**：入度刻画「还有几门先修课没修」，是「能不能开始」的判据；
出度刻画「学过它能解锁谁」，用来在修完后更新别人。

**代码**（`src/graph/course_schedule.py` / `.cpp`）：

```python
from collections import deque


def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indegree[course] += 1

    queue = deque(c for c in range(num_courses) if indegree[c] == 0)
    done = 0
    while queue:
        cur = queue.popleft()
        done += 1
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return done == num_courses
```

```cpp
bool canFinish(int numCourses, std::vector<std::vector<int>> &prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> indegree(numCourses, 0);
    for (auto &p : prerequisites) {
        int course = p[0], pre = p[1];
        graph[pre].push_back(course);
        ++indegree[course];
    }

    std::queue<int> q;
    for (int c = 0; c < numCourses; ++c) {
        if (indegree[c] == 0) q.push(c);
    }

    int done = 0;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        ++done;
        for (int nxt : graph[cur]) {
            if (--indegree[nxt] == 0) q.push(nxt);
        }
    }
    return done == numCourses;
}
```

- **复杂度**：时间 O(V + E)，空间 O(V + E)。
- **易错点**：建图方向别搞反——`prerequisites[i] = [a, b]` 是 `b -> a`，
  即先修 `b` 再修 `a`；只统计入度却忘了建邻接表，就无法在修课时更新别人；
  自环 `[0, 0]` 也会被正确判为有环；不要用「有没有课还剩」来判断，要用「出队个数」。
- **相似题**：210. 课程表 II（要具体顺序）；图上的有向环检测还有
  802. 找到最终安全状态、2050. 并行课程 III（拓扑 + DP）。

### 210. 课程表 II（中等）

**题目**：与 207 相同，但要求返回一种能修完所有课程的**学习顺序**；
若不可能修完（有环），返回空数组。

**思路**：
流程与 207 完全一样，只是这次的产物换成**出队顺序**：按出队的先后把课程记下来。
因为一门课只有在所有先修课都出队（入度降为 0）后才会出队，所以排在它前面的
必然是它的先修课，顺序天然合法。若最终顺序长度不足 `numCourses`，说明有环，
按题目要求返回空列表。

**为什么出队顺序就是合法拓扑序**：出队的时刻正是「所有入边（依赖）都已被删除」的时刻。

**代码**（`src/graph/course_schedule_ii.py` / `.cpp`）：

```python
from collections import deque


def find_order(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for course, pre in prerequisites:
        graph[pre].append(course)
        indegree[course] += 1

    queue = deque(c for c in range(num_courses) if indegree[c] == 0)
    order = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    return order if len(order) == num_courses else []
```

```cpp
std::vector<int> findOrder(int numCourses,
                           std::vector<std::vector<int>> &prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> indegree(numCourses, 0);
    for (auto &p : prerequisites) {
        int course = p[0], pre = p[1];
        graph[pre].push_back(course);
        ++indegree[course];
    }

    std::queue<int> q;
    for (int c = 0; c < numCourses; ++c) {
        if (indegree[c] == 0) q.push(c);
    }

    std::vector<int> order;
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        order.push_back(cur);
        for (int nxt : graph[cur]) {
            if (--indegree[nxt] == 0) q.push(nxt);
        }
    }

    if (static_cast<int>(order.size()) != numCourses) return {};
    return order;
}
```

- **复杂度**：时间 O(V + E)，空间 O(V + E)。
- **易错点**：多个入度为 0 的课谁先都一样，别去追求某个固定顺序（题目说任意一种即可）；
  判断有环的依据是「结果长度 != 节点数」，不是「队列为空」；
  返回空数组前别再输出半截顺序。
- **相似题**：207. 课程表（只问可行性）；两题共用同一套 Kahn 模板，
  差别只在「计数」还是「记录顺序」。

---

## 模式五：并查集，合并与查询连通块

**适用信号**：问题反复出现「把两个元素归到一组」「判断两个元素是否同组」，
典型如「有多少个连通块」「这条边是否多余」。当图的规模大到不适合每次重跑 BFS/DFS 时，
并查集几乎总是首选。

**两个核心操作**：

- `find(x)`：返回 `x` 所属集合的**代表元**（根）。两个元素同组 ⟺ 根相同；
- `union(a, b)`：把 `a`、`b` 所在的两个集合合并为一个。

**两个优化**：

- **路径压缩**：`find` 时顺手把沿途节点直接挂到根上（`parent[x] = parent[parent[x]]`），
  把树压扁，后续查找接近 O(1)；
- **按秩/按大小合并**（本文附赠思路）：总是把浅的树挂到深的树上，避免退化成链。
  两道题里路径压缩已足够，加上它会更稳。

### 547. 省份数量（中等）

**题目**：`n` 个城市，`isConnected[i][j] == 1` 表示城市 `i` 与 `j` 直接相连。
相连关系具有传递性，互相（直接或间接）连通的城市构成一个省份。返回省份数量。

**思路**：
「有几组互不相连的城市」就是「有几个连通块」，正好交给并查集。
初始时每个城市自成一个省份，省份数 `count = n`。遍历矩阵上三角，
凡是 `isConnected[i][j] == 1` 就尝试合并：

- 若两个城市本就同根，已在同一省份，跳过；
- 否则合并两个集合，**省份数减一**（两个省并成了一个）。

遍历结束 `count` 就是答案。

**为什么用「从 n 做减法」而不是最后去重**：每成功合并一次，独立集合就少一个，
直接得到省份数，省去再对每个城市 `find` 一遍去重。矩阵对称，
只遍历上三角即可，避免重复处理同一条边。

**代码**（`src/graph/find_provinces.py` / `.cpp`）：

```python
def find_circle_num(is_connected):
    n = len(is_connected)
    parent = list(range(n))
    count = n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j] == 1:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    count -= 1
    return count
```

```cpp
int findRoot(std::vector<int> &parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

int findCircleNum(std::vector<std::vector<int>> &isConnected) {
    int n = isConnected.size();
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    int count = n;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (isConnected[i][j] == 1) {
                int ri = findRoot(parent, i);
                int rj = findRoot(parent, j);
                if (ri != rj) {
                    parent[ri] = rj;
                    --count;
                }
            }
        }
    }
    return count;
}
```

- **复杂度**：时间 O(n²·α(n))，空间 O(n)。主要开销在遍历 `n x n` 矩阵，
  α(n) 是反阿克曼函数（实际可视为常数）。
- **易错点**：初始化 `parent[i] = i`（每个元素自成一类）不能漏；
  合并前必须先比较两个根，相同就别再减 `count`；
  只处理一条边就减一次，别重复计数；
  路径压缩写的是 `parent[x] = parent[parent[x]]`，别误改成把节点直接挂到 `x` 上。
- **相似题**：684. 冗余连接（判断一条边会不会成环）；
  200. 岛屿数量也可用并查集做（每个陆地格子与相邻陆地合并），但网格题用 DFS 更顺手。

### 684. 冗余连接（中等）

**题目**：一棵 `n` 个节点（编号 `1 ~ n`）的树被多加了一条边，形成唯一一个环。
给定边数组 `edges`，找出那条可以删掉的边，使剩下的图重新成为树；
若有多个答案，返回输入中最后出现的那条。

**思路**：
树的性质是「任意两点之间恰有一条路径、无环」。按输入顺序逐条把边加入并查集：

- 若边的两个端点当前**不在同一集合**，说明这条边把两个连通块连了起来，是必要的，
  执行 `union`；
- 若两端**已在同一集合**，说明它们之间本来就有路径，再加这条边就会成环——
  它正是那条多余的边。

**为什么返回的自动是「最后出现」的那条**：我们是从前往后逐条检查的，
一旦发现某条边会成环就立刻返回。题目保证只多了一条边、环唯一，
所以第一条（也是唯一一条）造成环的边就是答案；即使有多解，按输入顺序
检查得到的就是最后出现的那条。还有一个省事的等价判据：`n` 个节点的树应恰有
`n - 1` 条边，输入给了 `n` 条，所以 `n = len(edges)`，`parent` 开 `n + 1` 即可。

**代码**（`src/graph/redundant_connection.py` / `.cpp`）：

```python
def find_redundant_connection(edges):
    n = len(edges)
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return [u, v]
        parent[ru] = rv
    return []
```

```cpp
int findRoot(std::vector<int> &parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

std::vector<int> findRedundantConnection(std::vector<std::vector<int>> &edges) {
    int n = edges.size();
    std::vector<int> parent(n + 1);
    std::iota(parent.begin(), parent.end(), 0);

    for (auto &e : edges) {
        int u = e[0], v = e[1];
        int ru = findRoot(parent, u);
        int rv = findRoot(parent, v);
        if (ru == rv) return {u, v};
        parent[ru] = rv;
    }
    return {};
}
```

- **复杂度**：时间 O(n·α(n))（近似 O(n)），空间 O(n)。
- **易错点**：节点编号从 1 开始，`parent` 要开到 `n + 1`，下标别越界；
  必须**按输入顺序**处理，才能保证返回「最后出现」的那条；
  本题保证一定有一个环，但函数末尾仍返回空数组以保持完整；
  与 547 一样，合并前先比较根。
- **相似题**：547. 省份数量（统计连通块数）；685. 冗余连接 II（有向图版本，更复杂）。

---

## 模式六：反向思维，从边界向内标记

**适用信号**：题目要求处理「被包围的 / 内部的」区域，而「被包围」这个条件
从内部不容易判断，但从外部看一目了然。这时把问题**反过来**：先标记「不会被处理」
的那部分，剩下的就是要处理的。

### 130. 被围绕的区域（中等）

**题目**：`m x n` 矩阵 `board` 由 `'X'` 和 `'O'` 组成。捕获所有被 `'X'` 围绕的区域，
把其中的 `'O'` 全部改成 `'X'`。注意：边界上的 `'O'`，以及与边界 `'O'` 相连的 `'O'`，
都不会被填充。

**思路**：
直接判断某个 `'O'` 是否被完全包围很麻烦。反过来想：一个 `'O'` 不会被填充，
当且仅当它能沿四方向走到矩阵边界。于是：

1. 从**四条边界**上的每个 `'O'` 出发 DFS，把所有与边界相连的 `'O'` 临时标成 `'#'`，
   表示「安全、不可翻转」；
2. 收尾遍历整张矩阵：仍是 `'O'` 的说明被 `'X'` 完全包围，改成 `'X'`；
   是 `'#'` 的说明安全，恢复成 `'O'`。

**为什么从边界出发**：边界是「逃出去」的唯一出口。能从内部走到边界的 `'O'`
一定会被步骤 1 标记到；反之走不到边界的 `'O'` 就被困在内部，正是要翻转的对象。
这样就把「判断是否被包围」这个难题，换成了两次简单的连通块遍历。

**为什么用临时标记 `'#'`**：它同时承担两件事——记录「已访问」防止重复递归，
以及在收尾时区分「安全的 O」和「待翻转的 O」。相当于就地开了一个 visited 表。

**代码**（`src/graph/surrounded_regions.py` / `.cpp`）：

```python
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def solve(board):
    if not board or not board[0]:
        return
    rows, cols = len(board), len(board[0])

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
            return
        board[r][c] = "#"
        for dr, dc in _DIRS:
            dfs(r + dr, c + dc)

    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == "O":
                board[r][c] = "X"
            elif board[r][c] == "#":
                board[r][c] = "O"
```

```cpp
void dfs(std::vector<std::vector<char>> &board, int r, int c) {
    int rows = board.size();
    int cols = board[0].size();
    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != 'O') return;
    board[r][c] = '#';
    dfs(board, r + 1, c);
    dfs(board, r - 1, c);
    dfs(board, r, c + 1);
    dfs(board, r, c - 1);
}

void solve(std::vector<std::vector<char>> &board) {
    if (board.empty() || board[0].empty()) return;
    int rows = board.size();
    int cols = board[0].size();

    for (int r = 0; r < rows; ++r) {
        dfs(board, r, 0);
        dfs(board, r, cols - 1);
    }
    for (int c = 0; c < cols; ++c) {
        dfs(board, 0, c);
        dfs(board, rows - 1, c);
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c] == 'O') {
                board[r][c] = 'X';
            } else if (board[r][c] == '#') {
                board[r][c] = 'O';
            }
        }
    }
}
```

- **复杂度**：时间 O(m·n)，空间 O(m·n)（递归栈最坏情形）。
- **易错点**：只从左上角或某一条边出发会漏标记；四条边都要扫（角落会被扫两次，
  但第二次直接返回，无妨）；收尾时别把 `'#'` 漏还原成 `'O'`；
  空矩阵要先挡掉；DFS 递归太深时可改用 BFS，模板不变。
- **相似题**：200. 岛屿数量、695. 岛屿的最大面积（同为网格连通块，
  但这里从边界出发、做的是「反向标记」）；1254. 统计封闭岛屿的数目
  （同理，先把挨着边界的陆地淹掉，再数剩下的）。

---

## 规律总结

1. **网格就是隐式图，连通块问题用一次 DFS/BFS 解决**。节点是格子，边是上下左右邻接。
   「数个数」「求大小」「染色」「标记」这几种问法，共用同一套遍历骨架，只是入口处
   对遍历的利用方式不同：计数就计数、要大小就返回大小、要染色就改颜色。

2. **先想清楚「问的是什么」，再决定 DFS 还是 BFS**。
   - 问「有没有 / 有几个 / 多大 / 是否连通」——用 **DFS**，代码短；
   - 问「最短几步 / 最少几分钟 / 最近距离」——用 **BFS**，它天然按距离分层；
   - 问「合法的先后顺序 / 有没有循环依赖」——用 **拓扑排序**；
   - 只有「合并」和「是否同组」两种操作、且要反复做——用 **并查集**。

3. **DFS 模板只有四步**：判越界 → 判是否目标类型 → 标记已访问 → 递归邻接。
   顺序不能乱，越界判断必须最先做。网格上直接写四次递归比维护方向数组更直观。

4. **「标记」是防环的命门**。图有环（相邻节点互相指向），不标记已访问就会无限递归。
   就地修改输入（把 `'1'` 改 `'0'`、旧色改新色、`'O'` 改 `'#'`）是最省的标记方式，
   前提是允许修改；若要求保留原数据，就另开 visited 数组或哈希表。

5. **多源 BFS 的两个要点**：所有源在**第 0 层一次性**入队（相当于虚拟源点），
   以及**按层处理**（`len(queue)` 那层循环）才能正确累计「步数 / 分钟」。
   求每点到最近某物的距离、多个起点同时扩散，都归为这一类。

6. **拓扑排序（Kahn）模板**：统计入度 → 入度为 0 的入队 → 出队时把它指向的
   邻点入度减一、减到 0 就入队。结果长度等于节点总数 ⟺ 无环；
   要「顺序」就记录出队序列，要「可行性」就只比长度。

7. **并查集模板**：`parent[i] = i` 初始化，`find` 带路径压缩
   （`parent[x] = parent[parent[x]]`），`union` 前先比较根、同根则跳过。
   数连通块可以「从 n 开始，每合并一次减一」；判环可以「逐条加边，
   两端已同根就是多余的那条」。

8. **反向思维能化难为易**。「被 `'X'` 完全包围的 `'O'`」从内部难判断，
   就从**边界**出发标记「逃得出去的 `'O'`」，剩下的自然是要处理的。
   遇到「内部 / 被包围 / 封闭」这类词，先想想能不能从外面反向圈定。

9. **先把特例在循环外挡掉**：空网格/空图、起始色等于目标色、`best` 初值、
   队列为空……这些边界放在主逻辑之前处理，能让主干代码保持干净。
   深拷贝尤其要注意「先登记、再递归」，否则环上会无限递归。

10. **复杂度多为 O(V + E) 或 O(m·n)**。每个节点、每条边至多处理常数次；
    空间是 visited/并查集/队列的开销，DFS 还有 O(V) 的最坏递归深度。
    数据大又担心爆栈时，把 DFS 换成显式栈/队列的迭代写法，模板不变。

11. **`docs` 与 `src` 必须逐字一致**。题解代码与 `code/leetcode/src/graph/` 下的实现
    保持完全一致，以经过自测的 `src` 为准，文档只做粘贴。
