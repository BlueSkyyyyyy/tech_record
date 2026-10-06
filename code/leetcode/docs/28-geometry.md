# 计算几何：矩形、叉积、凸包与三维形体

几何题看着图形五花八门，真正用到的工具却很少：**向量叉积**、**两点距离**、
**哈希分组**、**凸包**。本专题这一篇把散落的经典几何题收拢成一条线：

1. **矩形相交**：一维区间求交 + 容斥，二维矩形就是两个方向各做一次。
2. **叉积**：一次乘减同时回答「共线吗」「在左边还是右边」「围出的面积多少」。
3. **距离矩阵**：四个点只看两两距离平方，就能判断是不是正方形。
4. **斜率哈希**：把方向用 gcd 约分成整数指纹，避免浮点误差。
5. **凸包**：单调链（Andrew）一趟求包围所有点的最小多边形。
6. **分组枚举**：把「同一中点和同一长度的对角线」分组，任意两条拼成一个矩形。

本篇收 10 道题，按六种套路分组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：矩形相交与面积 | 223. 矩形面积 / 836. 矩形重叠 | 中等 / 简单 |
| 模式二：叉积与距离判定 | 1037. 有效的回旋镖 / 593. 有效的正方形 | 简单 / 中等 |
| 模式三：三维形体的表面积与投影 | 892. 三维形体的表面积 / 883. 三维形体投影面积 | 中等 / 简单 |
| 模式四：斜率哈希——直线上最多的点 | 149. 直线上最多的点数 | 困难 |
| 模式五：凸包 | 587. 安装栅栏 | 困难 |
| 模式六：点集里找最小矩形 | 939. 最小面积矩形 / 963. 最小面积矩形 II | 中等 / 中等 |

> 几何题最容易踩的坑是**浮点数**。能用整数就算整数：叉积、距离平方、约分后的
> 方向对，都是精确的；只有面积/斜率这类本质上是「比值」的量才需要最后转浮点。
> 这与第 16 篇「位运算」强调的「用精确的整数运算代替反复试探」是同一种工程习惯。

---

## 模式一：矩形相交与面积

**适用信号**：题目出现「矩形重叠」「覆盖面积」「两个轴对齐的框」。

**核心动作**：把二维问题拆成 x、y 两个一维区间求交，再把两次结果相乘。
区间 `[a1, a2]` 与 `[b1, b2]` 的并集长度用 `max` 减 `min`，交集长度再取一个
`max(0, …)` 兜底。

### 223. 矩形面积（中等）

**题目**：给出两个轴对齐矩形的左下角与右上角坐标，返回它们覆盖的总面积
（重叠部分只算一次）。

**思路**：

```python
def compute_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    return area_a + area_b - width * height
```

```cpp
int computeArea(int ax1, int ay1, int ax2, int ay2,
                int bx1, int by1, int bx2, int by2) {
    int areaA = (ax2 - ax1) * (ay2 - ay1);
    int areaB = (bx2 - bx1) * (by2 - by1);
    int width = std::max(0, std::min(ax2, bx2) - std::max(ax1, bx1));
    int height = std::max(0, std::min(ay2, by2) - std::max(ay1, by1));
    return areaA + areaB - width * height;
}
```

**为什么是容斥**：两个矩形的并集面积 = 各自面积之和 − 重叠面积。重叠区域仍是
一个轴对齐矩形：它的左边界是两个左边界里靠右的那个（`max`），右边界是两个右边界
里靠左的那个（`min`），宽度就是它们的差；没有重叠时这个差为负，被 `max(0, …)`
直接压成 0，所以不用单独写 `if`。这正是第 04 篇「前缀和」容斥在几何里的同一招。

- **复杂度**：时间 O(1)，空间 O(1)。
- **易错点**：宽度、高度都要各自夹一次 `max(0, …)`，只夹一个方向会算错；
  面积相减不会溢出 32 位，但坐标范围大而乘积多时 C++ 里可用 `long long` 保险。
- **相似题**：836. 矩形重叠（只问是否相交，不求面积）。

### 836. 矩形重叠（简单）

**题目**：给定两个轴对齐矩形，判断它们是否重叠（交集面积大于 0）。

**思路**：

```python
def is_rectangle_overlap(rec1, rec2):
    x_overlap = min(rec1[2], rec2[2]) > max(rec1[0], rec2[0])
    y_overlap = min(rec1[3], rec2[3]) > max(rec1[1], rec2[1])
    return x_overlap and y_overlap
```

```cpp
bool isRectangleOverlap(std::vector<int>& rec1, std::vector<int>& rec2) {
    bool x = std::min(rec1[2], rec2[2]) > std::max(rec1[0], rec2[0]);
    bool y = std::min(rec1[3], rec2[3]) > std::max(rec1[1], rec2[1]);
    return x && y;
}
```

**为什么要严格大于**：矩形重叠 ⟺ 它在 x 轴上的投影相交、在 y 轴上的投影也相交。
一维区间相交的充要条件是「左端点的最大值 < 右端点的最小值」。用严格小于的
反面（即 `<` 写成 `>`）就把「只碰到一条边或一个点」排除掉了——那不算重叠。
如果写成 `>=`，相邻但不重叠的矩形会被误判。

- **复杂度**：时间 O(1)，空间 O(1)。
- **易错点**：两个方向都要判；边界相等（贴边）时必须返回 `false`。
- **相似题**：223. 矩形面积（同一套「区间求交」，还要算面积）。

---

## 模式二：叉积与距离判定

**适用信号**：判断「三点是否共线」「能否组成某个形状」「点在线段的哪一侧」。

**核心动作**：叉积 `cross(o,a,b) = (a-o) × (b-o) = (ax-ox)*(by-oy) -
(ay-oy)*(bx-ox)`。它等于以 o 为顶点、oa、ob 为两边的平行四边形**有向面积**：
- `cross > 0`：b 在 o→a 的左侧（逆时针）；
- `cross < 0`：b 在右侧（顺时针）；
- `cross == 0`：三点共线。

### 1037. 有效的回旋镖（简单）

**题目**：给定三个点，判断它们是否两两不同且不共线。

**思路**：

```python
def is_boomerang(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) != 0
```

```cpp
bool isBoomerang(std::vector<std::vector<int>>& points) {
    const auto& p1 = points[0];
    const auto& p2 = points[1];
    const auto& p3 = points[2];
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) -
               (p2[1] - p1[1]) * (p3[0] - p1[0]) !=
           0;
}
```

**为什么用叉积而不是斜率**：如果用斜率 `(y2-y1)/(x2-x1)` 判共线，遇到
「竖直线」（分母为 0）就得特判，还要处理浮点相等。叉积把它们统一成一次乘减，
不需要除法，也没有精度问题；`!= 0` 就表示三点不共线。题目保证三点互异，
所以只需判共线。

- **复杂度**：时间 O(1)，空间 O(1)。
- **易错点**：叉积公式的减号方向容易写反，但这里只看「是否为零」，
  正负不影响；真正要小心的是把 `x`、`y` 交叉写错。
- **相似题**：593. 有效的正方形（叉积/距离一起用）；149. 直线上最多的点数
  （叉积的「方向指纹」版本）。

### 593. 有效的正方形（中等）

**题目**：给定四个点，判断它们能否组成一个正方形。

**思路**：

```python
def valid_square(p1, p2, p3, p4):
    pts = [p1, p2, p3, p4]
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dists.append(dx * dx + dy * dy)
    dists.sort()
    side, diag = dists[0], dists[4]
    return (side > 0 and dists[0] == dists[1] == dists[2] == dists[3]
            and diag == dists[5] == 2 * side)
```

```cpp
bool validSquare(std::vector<int>& p1, std::vector<int>& p2,
                 std::vector<int>& p3, std::vector<int>& p4) {
    std::vector<std::vector<int>> pts{p1, p2, p3, p4};
    std::vector<long long> dists;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            long long dx = pts[i][0] - pts[j][0];
            long long dy = pts[i][1] - pts[j][1];
            dists.push_back(dx * dx + dy * dy);
        }
    }
    std::sort(dists.begin(), dists.end());
    long long side = dists[0];
    long long diag = dists[4];
    return side > 0 && dists[0] == dists[1] && dists[1] == dists[2] &&
           dists[2] == dists[3] && diag == dists[5] && diag == 2 * side;
}
```

**为什么只看六个距离**：四个点两两之间有 `C(4,2) = 6` 条线段，其中 4 条是边、
2 条是对角线。把距离平方排序后，正方形的特征非常干净：**前 4 个相等且大于 0**
（四条边等长、不退化成一点），**后 2 个相等且等于边长的 2 倍**（对角线平方 =
边平方 + 边平方）。用距离平方可以完全避开开根号和浮点比较。反过来，满足这组
条件的四点一定构成正方形，「边等 + 对角线符合勾股」正是正方形的判定。

- **复杂度**：时间 O(1)，空间 O(1)。
- **易错点**：`side > 0` 不能漏，否则四个重合点会被误判；比较的是**平方**，
  所以对角线条件是 `2 * side` 而不是 `side * sqrt(2)`。
- **相似题**：1037. 有效的回旋镖（不共线判定）。

---

## 模式三：三维形体的表面积与投影

**适用信号**：题目把「堆叠的方块」写成二维高度网格 `grid`，求表面积或投影。

**核心动作**：
- 表面积：每个柱子的上下底面固定贡献 2，四个侧面各贡献「比邻居高出的部分」
  `max(0, v - 邻居)`；
- 投影：三个坐标平面各是一个简单统计量，重叠只算一次，所以取 `max` 而不是求和。

### 892. 三维形体的表面积（中等）

**题目**：`grid[i][j]` 是位置 `(i, j)` 上堆叠的方块数，求整个立体的表面积。

**思路**：

```python
def surface_area(grid):
    n = len(grid)
    total = 0
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v == 0:
                continue
            total += 2
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                neighbor = grid[ni][nj] if 0 <= ni < n and 0 <= nj < n else 0
                if v > neighbor:
                    total += v - neighbor
    return total
```

```cpp
int surfaceArea(std::vector<std::vector<int>>& grid) {
    int n = static_cast<int>(grid.size());
    int total = 0;
    int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int v = grid[i][j];
            if (v == 0) {
                continue;
            }
            total += 2;
            for (auto& d : dirs) {
                int ni = i + d[0];
                int nj = j + d[1];
                int neighbor =
                    (ni >= 0 && ni < n && nj >= 0 && nj < n) ? grid[ni][nj] : 0;
                if (v > neighbor) {
                    total += v - neighbor;
                }
            }
        }
    }
    return total;
}
```

**为什么「差多少就差多少」**：把每个柱子的六个面分开算。上下两个底面永远露在
外面，贡献固定 2；朝某个方向的侧面，如果邻居没它高，露出的部分就是高度差
`v - neighbor`，如果邻居比它高（或一样高），这个方向完全被贴住，贡献 0。四个
方向独立相加即可，不用真的去搭三维模型。

- **复杂度**：时间 O(n^2)，空间 O(1)。
- **易错点**：越界方向的邻居高度按 0 处理；不要把 `neighbor` 也加成 `max(0, …)`，
  只在 `v > neighbor` 时加差值即可。
- **相似题**：883. 三维形体投影面积（同一网格、换种问法）。

### 883. 三维形体投影面积（简单）

**题目**：求立体在 xy、xz、yz 三个平面上的投影面积之和。

**思路**：

```python
def projection_area(grid):
    n = len(grid)
    top = sum(1 for row in grid for v in row if v > 0)
    front = sum(max(row) for row in grid)
    side = sum(max(grid[i][j] for i in range(n)) for j in range(n))
    return top + front + side
```

```cpp
int projectionArea(std::vector<std::vector<int>>& grid) {
    int n = static_cast<int>(grid.size());
    int top = 0, front = 0, side = 0;
    for (int i = 0; i < n; ++i) {
        int rowMax = 0;
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] > 0) {
                ++top;
            }
            rowMax = std::max(rowMax, grid[i][j]);
        }
        front += rowMax;
    }
    for (int j = 0; j < n; ++j) {
        int colMax = 0;
        for (int i = 0; i < n; ++i) {
            colMax = std::max(colMax, grid[i][j]);
        }
        side += colMax;
    }
    return top + front + side;
}
```

**为什么取 max 而不是求和**：投影是「从上往下压扁」，同一根柱子上的多个方块
在投影里只占一格，所以俯视只数「非零格子」；正视沿列看，一行里最高的那根柱子
决定了这一行投影的高度，故取该行最大值；侧视同理取每列最大值。三个方向互不
干扰，直接相加。

- **复杂度**：时间 O(n^2)，空间 O(1)。
- **易错点**：俯视是「有方块就 +1」而不是把高度加起来；正视与侧视一个按行、
  一个按列，别混了方向。
- **相似题**：892. 三维形体的表面积。

---

## 模式四：斜率哈希——直线上最多的点

**适用信号**：给一堆点，问「共线的最大点数」或「多少点在同一条直线上」。

**核心动作**：枚举一个点作基准，把到其余点的方向 `(dx, dy)` 用 gcd 约分成
最简整数对并统一符号，作为「方向指纹」放进哈希表计数。

### 149. 直线上最多的点数（困难）

**题目**：给定一组互不相同的点，求最多有多少个点在同一条直线上。

**思路**：

```python
from collections import defaultdict
from math import gcd


def max_points(points):
    n = len(points)
    if n <= 2:
        return n
    best = 0
    for i in range(n):
        xi, yi = points[i]
        directions = defaultdict(int)
        for j in range(i + 1, n):
            dx = points[j][0] - xi
            dy = points[j][1] - yi
            g = gcd(abs(dx), abs(dy)) or 1
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            directions[(dx, dy)] += 1
        if directions:
            best = max(best, 1 + max(directions.values()))
    return best
```

```cpp
int maxPoints(std::vector<std::vector<int>>& points) {
    int n = static_cast<int>(points.size());
    if (n <= 2) {
        return n;
    }
    int best = 0;
    for (int i = 0; i < n; ++i) {
        std::map<std::pair<int, int>, int> directions;
        for (int j = i + 1; j < n; ++j) {
            int dx = points[j][0] - points[i][0];
            int dy = points[j][1] - points[i][1];
            int g = std::gcd(std::abs(dx), std::abs(dy));
            if (g == 0) {
                g = 1;
            }
            dx /= g;
            dy /= g;
            if (dx < 0 || (dx == 0 && dy < 0)) {
                dx = -dx;
                dy = -dy;
            }
            ++directions[{dx, dy}];
        }
        for (const auto& kv : directions) {
            best = std::max(best, 1 + kv.second);
        }
    }
    return best;
}
```

**为什么必须约分 + 统一符号**：斜率 `dy/dx` 是「方向」的比值，`(1,2)` 与 `(2,4)`
表示同一个方向，但直接当键会被算成两条线，答案偏小。先用 `gcd` 把 `dx, dy`
约到互质，它们就成了方向的规范指纹；再把符号统一（规定 `dx > 0`，或
`dx == 0` 时 `dy > 0`），`(1,2)` 与 `(-1,-2)`（正好相反的方向）也会合并。
用整数对而不是浮点斜率，是为了让 `1/3` 与 `2/6` 一定相等，没有精度陷阱。
统计完后，过基准点 `i` 的某条线上共有「该方向计数 + 1」个点（加上 `i` 自己）。

- **复杂度**：时间 O(n^2 log C)（每对点一次 gcd，C 为坐标范围），空间 O(n)。
- **易错点**：只枚举 `j > i` 就够，方向是双向的；最后一轮基准点没有 `j > i`，
  哈希表可能为空，更新答案前要判空（C++ 里 `for` 空循环天然安全）；
  约分时 `gcd(0, 0)` 在本题不会出现（点互异），但写上 `or 1` 更稳。
- **相似题**：587. 安装栅栏（同样用叉积/方向处理点集）；939 与 963 也用
  「把几何关系翻译成哈希键」的思路。

---

## 模式五：凸包

**适用信号**：要求「包围所有点的最小多边形」「边界上的点」「围栏」。

**核心动作**：Andrew 单调链——按坐标排序，分别从左到右、从右到左扫描出下链、
上链，扫的过程中用叉积弹掉会造成「右转」的中间点。

### 587. 安装栅栏（困难）

**题目**：用最短的一圈栅栏把所有树围起来，返回落在栅栏边界上的所有树的坐标。

**思路**：

```python
def outer_trees(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    hull = set(lower) | set(upper)
    return [list(p) for p in sorted(hull)]
```

```cpp
long long cross(const std::pair<int, int>& o, const std::pair<int, int>& a,
                const std::pair<int, int>& b) {
    return 1LL * (a.first - o.first) * (b.second - o.second) -
           1LL * (a.second - o.second) * (b.first - o.first);
}

std::vector<std::vector<int>> outerTrees(std::vector<std::vector<int>> points) {
    std::vector<std::pair<int, int>> pts;
    for (const auto& p : points) {
        pts.push_back({p[0], p[1]});
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

    std::vector<std::vector<int>> result;
    if (pts.size() <= 2) {
        for (const auto& p : pts) {
            result.push_back({p.first, p.second});
        }
        return result;
    }

    std::vector<std::pair<int, int>> lower;
    for (const auto& p : pts) {
        while (lower.size() >= 2 &&
               cross(lower[lower.size() - 2], lower.back(), p) < 0) {
            lower.pop_back();
        }
        lower.push_back(p);
    }

    std::vector<std::pair<int, int>> upper;
    for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
        while (upper.size() >= 2 &&
               cross(upper[upper.size() - 2], upper.back(), *it) < 0) {
            upper.pop_back();
        }
        upper.push_back(*it);
    }

    std::set<std::pair<int, int>> hull(lower.begin(), lower.end());
    hull.insert(upper.begin(), upper.end());
    for (const auto& p : hull) {
        result.push_back({p.first, p.second});
    }
    return result;
}
```

**为什么两次扫描能拼出整圈**：把点按 `(x, y)` 排序后，凸包的边界天然分成
上半圈和下半圈。从左到右扫时，维护一条「始终向左拐」的折线，一旦出现右转
（叉积 < 0），说明中间那个点被包在内部，弹掉；扫完得到**下链**。从右到左
再用同一规则扫一遍，得到**上链**。两链并起来就是整圈凸包。

本题的特殊要求是「边界上共线的点也要保留」，所以判定条件是 **`< 0` 才弹**，
而不是常见的 `<= 0`：叉积等于 0 表示三点共线，此时中间点仍落在栅栏边上，必须
留下。去重后若点数不超过 2，它们本来就在边界上，直接返回。

- **复杂度**：排序 O(n log n)，两次扫描 O(n)，总体 O(n log n)；空间 O(n)。
- **易错点**：`< 0` 与 `<= 0` 的选择决定共线点保不保留，本题要 `< 0`；
  叉积在坐标大时用 64 位（C++ 里乘 `1LL`），下链与上链的端点会重复出现，
  最后要用集合去重；返回顺序任意，排序只是为了结果稳定。
- **相似题**：149. 直线上最多的点数（点集 + 叉积）；1453. 圆形靶内的最大
  飞镖数量（枚举圆心 + 距离判定）可作为进阶练习。

---

## 模式六：点集里找最小矩形

**适用信号**：给一堆点，问「能组成的最小矩形面积」。分两种：边平行于坐标轴
（939）与任意方向（963）。

**核心动作**：先把几何条件翻译成「哈希键」，再在同一个键对应的候选集合里两两
组合求面积。

### 939. 最小面积矩形（中等）

**题目**：求由 4 个点构成的、边平行于坐标轴的矩形的最小面积；不存在返回 0。

**思路**：

```python
from collections import defaultdict


def min_area_rect(points):
    by_x = defaultdict(list)
    for x, y in points:
        by_x[x].append(y)

    last = {}
    best = float("inf")
    for x in sorted(by_x):
        ys = sorted(by_x[x])
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                key = (ys[i], ys[j])
                if key in last:
                    area = (x - last[key]) * (ys[j] - ys[i])
                    if area < best:
                        best = area
                last[key] = x
    return 0 if best == float("inf") else best
```

```cpp
int minAreaRect(std::vector<std::vector<int>>& points) {
    std::map<int, std::vector<int>> byX;
    for (const auto& p : points) {
        byX[p[0]].push_back(p[1]);
    }

    std::map<std::pair<int, int>, int> last;
    int best = INT_MAX;
    for (auto& [x, ys] : byX) {
        std::sort(ys.begin(), ys.end());
        int m = static_cast<int>(ys.size());
        for (int i = 0; i < m; ++i) {
            for (int j = i + 1; j < m; ++j) {
                std::pair<int, int> key{ys[i], ys[j]};
                auto it = last.find(key);
                if (it != last.end()) {
                    int area = (x - it->second) * (ys[j] - ys[i]);
                    best = std::min(best, area);
                }
                last[key] = x;
            }
        }
    }
    return best == INT_MAX ? 0 : best;
}
```

**为什么记住「最近一次 x」就够**：轴对齐矩形由两条竖边确定，两条竖边必须横跨
同一段 `(y1, y2)`。把点按 `x` 分组后，每个组的 `y` 值两两都能组成一段竖直区间，
用 `(y1, y2)` 当键。从左到右扫：若这个键之前出现过，`last[key]` 就是左竖边的
`x`，当前 `x` 是右竖边，面积 = 宽 × 高。高固定时，宽越小面积越小，所以对每个键
只需保留**最近**的左边界——更早的左边界只会让宽更大，永远不会更优。

- **复杂度**：时间 O(n^2)（每个 x 组内枚举 y 对），空间 O(n^2)。
- **易错点**：每个 x 组内的 `y` 要排序，保证 `y1 < y2` 作为键；用
  `float("inf")` / `INT_MAX` 表示「还没有」；点集中同一列不会出现重复 `y`，
  否则要先去重。
- **相似题**：963. 最小面积矩形 II（允许任意方向）；149 也是「几何关系 → 哈希键」。

### 963. 最小面积矩形 II（中等）

**题目**：求由 4 个点构成的任意方向矩形的最小面积；不存在返回 0。

**思路**：

```python
from collections import defaultdict


def min_area_free_rect(points):
    n = len(points)
    groups = defaultdict(list)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            mid = (x1 + x2, y1 + y2)
            dist2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            groups[(mid, dist2)].append((i, j))

    best = float("inf")
    for pairs in groups.values():
        k = len(pairs)
        for a in range(k):
            i, j = pairs[a]
            d1x = points[j][0] - points[i][0]
            d1y = points[j][1] - points[i][1]
            for b in range(a + 1, k):
                p, q = pairs[b]
                d2x = points[q][0] - points[p][0]
                d2y = points[q][1] - points[p][1]
                area = abs(d1x * d2y - d1y * d2x)
                if area < best:
                    best = area
    return 0.0 if best == float("inf") else best / 2
```

```cpp
double minAreaFreeRect(std::vector<std::vector<int>>& points) {
    int n = static_cast<int>(points.size());
    std::map<std::tuple<int, int, long long>,
             std::vector<std::pair<int, int>>>
        groups;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            int mx = points[i][0] + points[j][0];
            int my = points[i][1] + points[j][1];
            long long dx = points[i][0] - points[j][0];
            long long dy = points[i][1] - points[j][1];
            long long dist2 = dx * dx + dy * dy;
            groups[{mx, my, dist2}].push_back({i, j});
        }
    }

    double best = -1.0;
    for (auto& [key, pairs] : groups) {
        int k = static_cast<int>(pairs.size());
        for (int a = 0; a < k; ++a) {
            int i = pairs[a].first, j = pairs[a].second;
            long long d1x = points[j][0] - points[i][0];
            long long d1y = points[j][1] - points[i][1];
            for (int b = a + 1; b < k; ++b) {
                int p = pairs[b].first, q = pairs[b].second;
                long long d2x = points[q][0] - points[p][0];
                long long d2y = points[q][1] - points[p][1];
                double area = std::abs(d1x * d2y - d1y * d2x) / 2.0;
                if (best < 0.0 || area < best) {
                    best = area;
                }
            }
        }
    }
    return best < 0.0 ? 0.0 : best;
}
```

**为什么分「中点 + 对角长度」的组**：一个四边形是矩形，当且仅当它的两条对角线
**互相平分且长度相等**。于是把每条候选对角线 `(i, j)` 按「中点的两倍坐标
`(x1+x2, y1+y2)` + 对角线长度的平方 `dx² + dy²`」分组。同一个组里任意两条
对角线都有相同的中点和长度，拼起来一定是一个矩形（它们都是同一个圆的两条直径）。
两条对角线向量叉积的绝对值 = 以它们为对角线的四边形面积的 2 倍，所以面积是
`|叉积| / 2`。枚举组内两两组合取最小即可。中点用坐标和（两倍）表示，长度用平方，
全程整数；只有最后面积是「比值」才转浮点。

- **复杂度**：时间 O(n^2)（枚举点对 + 组内组合），空间 O(n^2)。
- **易错点**：中点的键用 `x1+x2, y1+y2`（两倍）而不是除以 2，避免小数；
  同一组里两条对角线不会共用端点（否则它们会完全相同），所以四点一定互异；
  没有矩形时返回 `0.0`，注意 Python 里 `float("inf")` 的判断与 C++ 里
  用 `-1.0` 当哨兵的区别。
- **相似题**：939. 最小面积矩形（轴对齐版本）；587/149 都在用叉积处理点集。

---

## 规律总结

1. **能用整数就别用浮点**。叉积、距离平方、约分后的方向对都是精确整数；只有
   面积、斜率这类本质是比值的量才在最后转浮点。593 用距离平方代替边长、149 用
   gcd 约分代替斜率、963 用「两倍中点」代替中点，都是同一思路。

2. **二维几何问题先拆成一维**。矩形相交 = x、y 两个区间分别相交（836），
   并集面积 = 两块面积相加减交集（223）。把二维的 `min/max` 拆开，条件就变得
   一目了然。

3. **叉积是几何题的瑞士军刀**：判共线（1037）、判方向、求四边形面积（963）、
   求凸包（587）都靠它。记住 `cross > 0` 是逆时针（左转）、`cross < 0` 是
   顺时针（右转）、`cross == 0` 是共线，正负号本身也携带信息。

4. **凸包用单调链，两趟扫描**。排序后从左到右求下链、从右到左求上链；「弹掉
   右转点」是核心动作。要不要保留共线点，取决于判定用 `<= 0` 还是 `< 0`
   （587 要求保留，故用 `< 0`）。

5. **几何关系翻译成哈希键**，是「点集里找特定图形」的通用套路：方向指纹
   （149）、竖直区间（939）、中点和长度（963）。翻译得好，问题就退化成「在
   同一组候选里找最优」。

6. **三维网格题往往只是二维统计**。表面积 = 上下底面 + 各方向高差（892），
   投影 = 三个方向的计数/最大值（883）。不用真的构造立方体，逐格做局部统计
   即可。

7. **与其它篇的联系**：223 的容斥就是第 04 篇「前缀和与差分」里二维前缀和的
   加减思路；「几何关系 → 哈希键」与第 02 篇「哈希表」、第 03 篇「滑动窗口」
   的「把状态压成一个键」同源；凸包虽然属于计算几何，但扫描时用到的单调
   「弹栈」动作与第 07 篇「栈与单调栈」如出一辙。
