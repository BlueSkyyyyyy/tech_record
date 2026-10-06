# 图与搜索：网格 DFS / BFS 入门

「图」听起来抽象，但面试里最常见的图根本不用建邻接表——**网格（二维数组）本身就是一张图**。
每个格子是一个节点，上下左右相邻就有一条边。岛屿、迷宫、染色、连通区域这类题，
都是在问「这块网格里有几个连通块」「某一块有多大」，用一次 **DFS 或 BFS 把相连的格子走一遍**
就能解决。

本篇先建立最基本的一套「网格搜索」思维：**从某个格子出发，沿着四方向把整个连通块标记掉**。
后面图论专题（拓扑排序、并查集、最短路）会在这套模板上继续加内容，所以这里把地基打牢。

网格搜索有两个恒定的小零件，先记住：

- **方向**：四方向偏移 `(-1,0) (1,0) (0,-1) (0,1)`，或者直接写四次递归；
- **越界判断**：`r < 0 || r >= rows || c < 0 || c >= cols` 必须写在最前面，
  否则会访问到非法内存（C++ 直接崩溃）。

本篇题目（由易到难，同一模板的三种用法）：

| 模式 | 题目 | 难度 |
|---|---|---|
| 网格 DFS：标记连通块 | 200. 岛屿数量 | 中等 |
| 网格 DFS：标记连通块 | 733. 图像渲染 | 简单 |
| 网格 DFS：标记连通块 | 695. 岛屿的最大面积 | 中等 |

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

## 规律总结

1. **网格就是隐式图，连通块问题用一次 DFS/BFS 解决**。节点是格子，边是上下左右邻接。
   「数个数」「求大小」「染色」「标记」这几种问法，共用同一套遍历骨架，只是入口处
   对 DFS 的利用方式不同：计数就计数、要大小就返回大小、要染色就改颜色。

2. **DFS 模板只有四步**：判越界 → 判是否目标类型 → 标记已访问 → 递归四个方向。
   顺序不能乱，越界判断必须最先做。写得越简单越不容易错，四个方向直接写四次递归
   比维护方向数组更直观。

3. **「标记」是防环的命门**。网格图有环（相邻格子互相指向），不标记已访问就会无限递归。
   就地修改输入（把 `'1'` 改 `'0'`、把旧色改新色）是最省的标记方式，前提是允许修改；
   若要求保留原数据，就另开 visited 数组。

4. **先把特例在循环外挡掉**：空网格、起始色等于目标色、`best` 初值……这些边界
   放在主逻辑之前处理，能让主干代码保持干净，也避免把特判塞进递归里。

5. **复杂度恒为 O(m·n)（时间）与 O(m·n)（最坏栈空间）**。每个格子至多进出一次；
   递归深度最坏是整片连通块的大小。若数据规模大又担心爆栈，把 DFS 换成
   显式栈/队列的迭代写法即可，模板不变。

6. **DFS 与 BFS 在网格上多数时候可互换**。DFS 代码短、适合求连通块大小/染色；
   BFS 天然按「距离」分层，适合求「最近」「最短步数」。等图专题遇到「腐烂的橘子」
   「01 矩阵」这类求最短距离的题时，就该切到 BFS。

7. **`docs` 与 `src` 必须逐字一致**。题解代码与 `code/leetcode/src/graph/` 下的实现
   保持完全一致，以经过自测的 `src` 为准，文档只做粘贴。
