# 矩阵与二维数组：原地变换、螺旋与结构化查找

二维数组（矩阵）题看似零散，其实反复出现的只有几个动作：**交换下标、按边界走、
按对角线走、判断相邻关系、原地打标记**。它们的共同难点不在算法复杂度，而在
「**下标别写错、原地修改别把数据提前破坏掉**」。很多题的最优解都要求原地完成，
这就逼着我们把矩阵本身当成存储，用它的行、列、某个二进制位来暂存中间信息。

本篇按由易到难铺开十个模式。前面几题（867、566、48）是纯下标练习，讲清「转置」
「旋转」「重塑」这些变换的坐标公式；中间几题（59、498、766）训练按特定顺序遍历
矩阵；后面几题（73、289、36、240）则各自展示一种「借矩阵自己当工作内存」或
「利用有序性排除一半搜索空间」的技巧。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：下标交换与形状变换 | 867. 转置矩阵 / 566. 重塑矩阵 | 简单 |
| 模式二：原地旋转（转置 + 翻转） | 48. 旋转图像 | 中等 |
| 模式三：按边界螺旋填充 | 59. 螺旋矩阵 II | 中等 |
| 模式四：按「行列和」枚举对角线 | 498. 对角线遍历 | 中等 |
| 模式五：判断相邻关系 | 766. 托普利茨矩阵 | 简单 |
| 模式六：借首行首列当标记 | 73. 矩阵置零 | 中等 |
| 模式七：用第二个二进制位存中间态 | 289. 生命游戏 | 中等 |
| 模式八：三组集合做约束校验 | 36. 有效的数独 | 中等 |
| 模式九：有序矩阵的阶梯查找 | 240. 搜索二维矩阵 II | 中等 |

读这一篇时，请把注意力放在**坐标变换**和**原地技巧**两件事上。坐标变换的题，先在
纸上写出「原位置 `(i, j)` 应该去哪个新位置」，公式对了代码就是抄；原地题的通用
思路是「先想清楚会不会覆盖后面还要用的数据」，若会，就想办法用额外的几个变量、
或者借用矩阵自己的某一行/某一列/某一位来保存。

---

## 模式一：下标交换与形状变换

**适用信号**：要求转置、翻转、重塑（reshape）矩阵；核心是找到「旧下标 → 新下标」
的映射公式，逐元素搬运即可。

**核心动作**：转置是 `(i, j) → (j, i)`，是把行列下标互换；重塑是保持行优先顺序、
只改变每行放几个，用一维序号 `k` 当桥梁。

### 867. 转置矩阵（简单）

**题目**：给定 m x n 矩阵 `matrix`，返回它的转置矩阵（n x m），满足
`result[j][i] == matrix[i][j]`。

**思路**：

转置的本质就是交换行列下标。Python 里最直接的写法是 `zip(*matrix)`：`*matrix`
把每一行摊开成参数，`zip` 会把各行的第 0 个、第 1 个、……元素分别打包，正好构成
转置后的每一行；`zip` 返回元组，再转成 `list` 即可。

**为什么可以这么写**：`zip` 的第 `j` 次产出，收集的是「每一行的第 `j` 个元素」，
按行号排好就是原矩阵的第 `j` 列——而第 `j` 列正是转置矩阵的第 `j` 行。这就是一次
「行列互换」。C++ 里没有 `zip`，直接双重循环写 `result[j][i] = matrix[i][j]`，
是同一件事的展开版本，两者都掌握最好。

**代码**（完整可运行版见 `src/matrix/transpose.py` / `.cpp`）：

```python
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]
```

```cpp
std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    std::vector<std::vector<int>> result(n, std::vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}
```

- **复杂度**：时间 O(m*n)（每个元素搬一次），空间 O(m*n)（返回结果本身）。
- **易错点**：转置后行列数互换，新矩阵是 `n` 行 `m` 列，别写反；非方阵绝不能
  原地转置（会越界），必须新开矩阵；`zip(*matrix)` 在矩阵为空列表时会报错，
  边界情况要先判空。
- **相似题**：48. 旋转图像（方阵版的原地转置 + 翻转，见模式二）；
  566. 重塑矩阵（同样是把元素按行优先顺序重新摆放，见下）；
  867 若要「原地」可对正方形矩阵做交换 `(i,j)↔(j,i)`。

### 566. 重塑矩阵（简单）

**题目**：给定 m x n 矩阵 `mat` 和目标形状 `r x c`。若元素总数相同（`m*n == r*c`），
把它按「逐行展开」顺序重新排成 `r` 行 `c` 列；否则原样返回。

**思路**：

重塑前后元素的「行优先顺序」完全一致，只是每行放几个变了。把一个元素在一维
数组里的序号记为 `k`（从 0 数到 `m*n-1`），它和两种形状下的二维坐标关系是：

```
原矩阵：i = k // n, j = k % n
新矩阵：i' = k // c, j' = k % c
```

遍历 `k`，用第一个公式从原处取出，用第二个公式放进新处即可。

**为什么先判元素总数**：总数对不上时目标形状根本不存在，题目要求原样返回。这是
本题最常被忽略的边界。另外别忘了前提是「行优先」展开，不是按列。

**代码**（`src/matrix/reshape_matrix.py` / `.cpp`）：

```python
def matrix_reshape(mat, r, c):
    m, n = len(mat), len(mat[0])
    if m * n != r * c:
        return mat
    res = [[0] * c for _ in range(r)]
    for k in range(m * n):
        res[k // c][k % c] = mat[k // n][k % n]
    return res
```

```cpp
std::vector<std::vector<int>> matrixReshape(const std::vector<std::vector<int>> &mat,
                                            int r, int c) {
    int m = static_cast<int>(mat.size());
    int n = static_cast<int>(mat[0].size());
    if (m * n != r * c) {
        return mat;
    }
    std::vector<std::vector<int>> res(r, std::vector<int>(c, 0));
    for (int k = 0; k < m * n; ++k) {
        res[k / c][k % c] = mat[k / n][k % n];
    }
    return res;
}
```

- **复杂度**：时间 O(m*n)，空间 O(r*c)（新矩阵；直接复用 `mat` 的题可做到 O(1)
  额外空间，但通常没必要）。
- **易错点**：`k // c`、`k % c` 用的是新列数 `c`，`k // n`、`k % n` 用的是旧列数
  `n`，很容易混；忘记判 `m*n != r*c` 直接构造会越界或丢元素。
- **相似题**：867. 转置矩阵（见上）；118. 杨辉三角（`docs/13-dynamic-programming.md`
  里「结果本身就是一张表」的典型）；304. 二维区域和检索
  （`docs/04-prefix-sum.md`，也是在一个二维结构上按坐标存取）。

---

## 模式二：原地旋转（转置 + 翻转）

**适用信号**：要求把 n x n 方阵原地顺时针/逆时针旋转 90 度、180 度。

**核心动作**：把一次旋转拆成两个「只交换两格」的小动作——先沿主对角线转置，
再对每行做水平翻转（顺时针）；逆时针则改成转置 + 垂直翻转。

### 48. 旋转图像（中等）

**题目**：给定 n x n 矩阵 `matrix`，原地把它顺时针旋转 90 度。

**思路**：

把「顺时针旋转 90 度」拆成两步：

1. 沿主对角线转置：交换 `matrix[i][j]` 与 `matrix[j][i]`；
2. 对每一行做水平翻转（左右颠倒）。

**为什么恰好是这两步**：设原坐标 `(i, j)`。转置后到 `(j, i)`，再水平翻转把列坐标
变成 `n-1-i`，即 `(j, n-1-i)`——这正是顺时针 90 度的坐标变换结果。分解成两步的
好处是每一步都只是「交换两个元素」，下标不容易写错；而一次性的「四元环交换」
虽然只走一趟，但边界处理更绕。

**代码**（`src/matrix/rotate_image.py` / `.cpp`）：

```python
def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
```

```cpp
void rotate(std::vector<std::vector<int>> &matrix) {
    int n = static_cast<int>(matrix.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }
    for (int i = 0; i < n; ++i) {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}
```

- **复杂度**：时间 O(n²)（转置与翻转各遍历约一半/全部元素），空间 O(1)（原地）。
- **易错点**：转置的内层循环必须从 `j = i + 1` 开始，从 0 开始会把每对换两次又
  换回去；「转置 + 水平翻转」是顺时针，「转置 + 垂直翻转」是逆时针，别配错；
  翻转是「整行 reverse」，不是和首尾行交换。
- **相似题**：867. 转置矩阵（本解的第一步，见模式一）；54. 螺旋矩阵与
  59. 螺旋矩阵 II（`docs/01-array-two-pointers.md` / 本篇模式三，都是绕着方阵走）；
  面试题 01.07 旋转矩阵（就是本题）。

---

## 模式三：按边界螺旋填充

**适用信号**：要求按螺旋顺序读/写矩阵，或生成螺旋矩阵。

**核心动作**：维护 `top / bottom / left / right` 四条边界，每走完一个方向就把对应
边界向内收一格；走完四条边就进入下一圈。

### 59. 螺旋矩阵 II（中等）

**题目**：给定正整数 n，生成一个 n x n 方阵，按顺时针从外到内螺旋填充 1 到 n*n。

**思路**：

维护四条边界，每一轮依次：

- 从左到右填 `top` 行，填完 `top += 1`；
- 从上到下填 `right` 列，填完 `right -= 1`；
- 若还有行，从右到左填 `bottom` 行，填完 `bottom -= 1`；
- 若还有列，从下到上填 `left` 列，填完 `left += 1`。

**为什么后两个方向要再判一次 `top <= bottom` / `left <= right`**：当 n 为奇数时，
最内圈可能只剩单独一行（或一列）。如果不判断就继续填，会把刚填过的位置反向覆盖。
加上这两个判断，循环就严格在「还有空间」时才走。

本题是「螺旋矩阵」（54 题）的生成版：54 是把已有矩阵按螺旋顺序读出来，用的也是
这组边界，只是把「写值」换成了「读值」。

**代码**（`src/matrix/spiral_matrix_ii.py` / `.cpp`）：

```python
def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1
    return matrix
```

```cpp
std::vector<std::vector<int>> generateMatrix(int n) {
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    int top = 0, bottom = n - 1, left = 0, right = n - 1;
    int num = 1;
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; ++j) {
            matrix[top][j] = num++;
        }
        ++top;
        for (int i = top; i <= bottom; ++i) {
            matrix[i][right] = num++;
        }
        --right;
        if (top <= bottom) {
            for (int j = right; j >= left; --j) {
                matrix[bottom][j] = num++;
            }
            --bottom;
        }
        if (left <= right) {
            for (int i = bottom; i >= top; --i) {
                matrix[i][left] = num++;
            }
            ++left;
        }
    }
    return matrix;
}
```

- **复杂度**：时间 O(n²)（每个格子填一次），空间 O(n²)（结果矩阵本身）。
- **易错点**：四个方向的边界闭区间要写对（`range(left, right + 1)` 是闭的）；每填
  完一条边立刻收缩对应边界；后两条边一定要用 `if` 保护，否则奇数 n 会重复覆盖；
  初始化矩阵时不要用 `[[0]*n]*n`（那是 n 个相同引用），要用列表推导。
- **相似题**：54. 螺旋矩阵（`docs/01-array-two-pointers.md`，读取版，同一边界法）；
  48. 旋转图像（本篇模式二，也是绕方阵走）；885. 螺旋矩阵 III（按坐标走、越界才
  转向，边界法不适用，改为模拟）。

---

## 模式四：按「行列和」枚举对角线

**适用信号**：要求按对角线方向遍历矩阵，且相邻对角线方向交替。

**核心动作**：同一条对角线上的元素满足 `i + j = d`，`d` 从 0 到 `m+n-2`；`d` 的
奇偶天然决定方向，逐条对角线走即可。

### 498. 对角线遍历（中等）

**题目**：给定 m x n 矩阵 `mat`，按对角线顺序遍历所有元素，方向交替（第 1 条向
右上，第 2 条向左下，第 3 条向右上……），返回遍历序列。

**思路**：

第 `d` 条对角线上的元素都满足 `i + j = d`，`d` 从 0 取到 `m+n-2`，共 `m+n-1` 条。
对每条对角线，先算出进入矩阵的起点，再顺方向走：

- `d` 为偶数：从左下往右上走；
- `d` 为奇数：从右上往左下走。

起点要保证不越界：向上走时取 `r = min(d, m-1)`、`c = d - r`；向下走时取
`c = min(d, n-1)`、`r = d - c`。

**为什么用 `d` 而不是逐格判断方向**：`d` 的奇偶天然对应方向，每条对角线彼此独立，
不需要维护「上一步往哪走」这种全局状态，边界条件更少，也更不容易出错。

**代码**（`src/matrix/diagonal_traverse.py` / `.cpp`）：

```python
def find_diagonal_order(mat):
    m, n = len(mat), len(mat[0])
    res = []
    for d in range(m + n - 1):
        if d % 2 == 0:
            r = min(d, m - 1)
            c = d - r
            while r >= 0 and c < n:
                res.append(mat[r][c])
                r -= 1
                c += 1
        else:
            c = min(d, n - 1)
            r = d - c
            while c >= 0 and r < m:
                res.append(mat[r][c])
                r += 1
                c -= 1
    return res
```

```cpp
std::vector<int> findDiagonalOrder(const std::vector<std::vector<int>> &mat) {
    int m = static_cast<int>(mat.size());
    int n = static_cast<int>(mat[0].size());
    std::vector<int> res;
    for (int d = 0; d < m + n - 1; ++d) {
        if (d % 2 == 0) {
            int r = std::min(d, m - 1);
            int c = d - r;
            while (r >= 0 && c < n) {
                res.push_back(mat[r][c]);
                --r;
                ++c;
            }
        } else {
            int c = std::min(d, n - 1);
            int r = d - c;
            while (c >= 0 && r < m) {
                res.push_back(mat[r][c]);
                ++r;
                --c;
            }
        }
    }
    return res;
}
```

- **复杂度**：时间 O(m*n)（每个元素恰好访问一次），空间 O(m*n)（返回序列）。
- **易错点**：对角线条数是 `m+n-1` 而不是 `max(m, n)`；起点用 `min` 夹住，否则在
  非方阵里第一步就出界；`d` 的奇偶与方向对应关系写反会得到反向的遍历序列；每个
  `while` 的两个越界条件都要带（行和列各一个）。
- **相似题**：54. 螺旋矩阵、59. 螺旋矩阵 II（本篇模式三，另一种「非行非列」的
  遍历顺序）；1424. 对角线遍历 II（按对角线分组后排序输出，是本题的进阶版）。

---

## 模式五：判断相邻关系

**适用信号**：要检查矩阵中某种「沿某个方向相同/递增/对称」的性质。

**核心动作**：只比较每个元素与它**前一个方向的邻居**，利用传递性判断整条线。

### 766. 托普利茨矩阵（简单）

**题目**：若矩阵每一条从左上到右下的对角线上的元素都相同，就称它是托普利茨矩阵。
给定矩阵，判断它是否满足。

**思路**：

对每个非首行、非首列的元素 `matrix[i][j]`，它都应该和左上角的
`matrix[i-1][j-1]` 相等。一旦有不等立即返回 `False`，全部通过则为 `True`。

**为什么只比左上邻居就够**：同一条对角线上的元素满足 `i - j` 相同，相邻两个位置
正好是 `(i-1, j-1)` 与 `(i, j)`。只要每一对相邻都相等，由相等的传递性，整条对角线
自然都相等，不必两两比较。

**代码**（`src/matrix/toeplitz_matrix.py` / `.cpp`）：

```python
def is_toeplitz_matrix(matrix):
    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):
            if matrix[i][j] != matrix[i - 1][j - 1]:
                return False
    return True
```

```cpp
bool isToeplitzMatrix(const std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][j] != matrix[i - 1][j - 1]) {
                return false;
            }
        }
    }
    return true;
}
```

- **复杂度**：时间 O(m*n)，空间 O(1)。
- **易错点**：循环从下标 1 开始（下标 0 没有左上邻居）；比的是 `(i-1, j-1)`，写成
  `(i-1, j)` 或 `(i, j-1)` 就变成「每行相同」或「每列相同」，是另外的题；
  单行或单列矩阵天然是托普利茨矩阵（循环不执行，返回 `True`）。
- **相似题**：240. 搜索二维矩阵 II（本篇模式九，也依赖行、列各自的单调性）；
  74. 搜索二维矩阵（`docs/05-binary-search.md`，把有序矩阵展平后二分）；
  378. 有序矩阵中第 K 小的元素（`docs/08-heap.md`，多路归并）。

---

## 模式六：借首行首列当标记

**适用信号**：要求原地修改矩阵，但需要记录「哪些行/列被触发」，又不想用 O(m+n)
额外空间。

**核心动作**：把首行、首列当成标记位；它们自己的归属信息单独用两个变量保存。

### 73. 矩阵置零（中等）

**题目**：给定 m x n 矩阵 `matrix`，若某元素为 0，则把它的整行和整列都置 0。
要求原地修改。

**思路**：

不能一遇到 0 就立刻清零整行整列，那会污染后面还没读的格子。正确做法分三步：

1. 用变量 `col0` 记下「第 0 列本身是否含 0」；再扫描除第 0 行、第 0 列以外的所有
   格子，遇到 0 就在首行/首列做标记：`matrix[i][0] = 0`、`matrix[0][j] = 0`。
2. 再扫一遍这些格子，只要所在行的标记 `matrix[i][0]` 或所在列的标记
   `matrix[0][j]` 为 0，就把当前位置 0。
3. 最后补第 0 行、第 0 列：`matrix[0][0] == 0` 说明第 0 行要清零，`col0` 为真说明
   第 0 列要清零。

**为什么能省到 O(1) 空间**：首行和首列本身就是天然的「这一行/列要不要清零」的
备忘录。代价是它们自己的信息得先挪走：第 0 行用 `matrix[0][0]` 记，第 0 列用
`col0` 记。这样标记与正文共用同一块存储，额外空间只有 O(1)。

**代码**（`src/matrix/set_matrix_zeroes.py` / `.cpp`）：

```python
def set_zeroes(matrix):
    m, n = len(matrix), len(matrix[0])
    col0 = any(matrix[i][0] == 0 for i in range(m))
    for i in range(m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if matrix[0][0] == 0:
        for j in range(n):
            matrix[0][j] = 0
    if col0:
        for i in range(m):
            matrix[i][0] = 0
```

```cpp
void setZeroes(std::vector<std::vector<int>> &matrix) {
    int m = static_cast<int>(matrix.size());
    int n = static_cast<int>(matrix[0].size());

    bool col0 = false;
    for (int i = 0; i < m; ++i) {
        if (matrix[i][0] == 0) {
            col0 = true;
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                matrix[i][j] = 0;
            }
        }
    }
    if (matrix[0][0] == 0) {
        for (int j = 0; j < n; ++j) {
            matrix[0][j] = 0;
        }
    }
    if (col0) {
        for (int i = 0; i < m; ++i) {
            matrix[i][0] = 0;
        }
    }
}
```

- **复杂度**：时间 O(m*n)，空间 O(1)（只用常数个额外变量）。
- **易错点**：`matrix[0][0]` 同时被用作「第 0 行有 0」的标记，但它也可能是第 0 列
  的标记，所以第 0 列必须另用 `col0` 保存，两者不能混；正文扫描要从 `j = 1` 开始
  （首列单独处理）；第 0 行、第 0 列的清理必须放在最后，否则会提前污染标记。
- **相似题**：289. 生命游戏（本篇模式七，同样要求原地、同样靠暂存中间信息）；
  36. 有效的数独（本篇模式八，也用到「用行/列做约束记录」的思想）；
  867 转置矩阵（如果允许 O(m*n) 额外空间，置零可以新开矩阵，简单但费空间）。

---

## 模式七：用第二个二进制位存中间态

**适用信号**：模拟类原地题，新状态依赖所有邻居的旧状态，不能边读边覆盖。

**核心动作**：把「旧状态」放最低位、「新状态」放次低位；读邻居时只读最低位，
全部算完后整体右移一位，新状态就转正。

### 289. 生命游戏（中等）

**题目**：给定 m x n 的 0/1 棋盘（1 表示活细胞），按规则原地更新：活细胞周围恰有
2 或 3 个活细胞时继续存活，否则死亡；死细胞周围恰有 3 个活细胞时复活。

**思路**：

难点是「算某格新状态时不能破坏邻居的旧状态」。把两种状态放进同一个整数的不同
二进制位：最低位存当前状态（读邻居时只取这一位），次低位存下一步状态。这样写
中间状态不会覆盖旧状态，邻居之间互不干扰。全部算完后统一右移一位即可。

编码：`0b10` 表示「现在死、下一步活」；`0b01` 表示「现在活、下一步死」；`0b00`、
`0b11` 表示状态不变。右移后 `0b11→1`、`0b10→1`、`0b01→0`、`0b00→0`。

**代码**（`src/matrix/game_of_life.py` / `.cpp`）：

```python
def game_of_life(board):
    m, n = len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            live = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n:
                        live += board[ni][nj] & 1
            if board[i][j] & 1:
                board[i][j] = 0b11 if live in (2, 3) else 0b01
            else:
                board[i][j] = 0b10 if live == 3 else 0b00
    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1
```

```cpp
void gameOfLife(std::vector<std::vector<int>> &board) {
    int m = static_cast<int>(board.size());
    int n = static_cast<int>(board[0].size());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int live = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    if (di == 0 && dj == 0) {
                        continue;
                    }
                    int ni = i + di;
                    int nj = j + dj;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                        live += board[ni][nj] & 1;
                    }
                }
            }
            if (board[i][j] & 1) {
                board[i][j] = (live == 2 || live == 3) ? 0b11 : 0b01;
            } else {
                board[i][j] = (live == 3) ? 0b10 : 0b00;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            board[i][j] >>= 1;
        }
    }
}
```

- **复杂度**：时间 O(m*n)（每格固定看 8 个邻居），空间 O(1)（原地）。
- **易错点**：统计邻居时必须写 `board[ni][nj] & 1`，只取旧状态位，否则会把刚写进去
  的新状态也算进去；八个方向要跳过自己 `(di, dj) == (0, 0)`；越界判断不能漏；
  最后一定要统一右移，否则返回的是编码后的中间值。
- **相似题**：73. 矩阵置零（本篇模式六，原地 + 暂存）；36. 有效的数独（同为原地
  约束检查，见下）；这类「用额外位/标记暂存新状态」的套路在状态压缩 DP 中也常见
  （`docs/13-dynamic-programming.md` 的状压思想）。

---

## 模式八：三组集合做约束校验

**适用信号**：要检查矩阵里每个位置是否满足「行列/宫」等分组内的唯一性约束。

**核心动作**：为每一类分组各维护一个集合，边遍历边查重、边插入；分组编号用
整除与取模算出来。

### 36. 有效的数独（中等）

**题目**：给定 9x9 数独盘面，判断已填数字是否违反规则：每行、每列、每个 3x3 宫内
数字 1-9 不重复。空格用 `.` 表示，不需要判断是否有解。

**思路**：

为每一行、每一列、每一个 3x3 宫各维护一个集合。遍历每个已填数字：

- 若它已在行集合 `rows[i]`、列集合 `cols[j]` 或宫集合 `boxes[k]` 中，返回 `False`；
- 否则把它同时加入这三个集合。

宫殿编号用 `k = (i // 3) * 3 + j // 3`：把 9 个宫按行优先编号 0-8，这个式子用
「行块号 * 3 + 列块号」拼出编号，正好一一对应。

**为什么用集合而不是排序比较**：集合的插入、查找都是 O(1)，一次遍历即可，还能在
发现冲突的第一时间退出；空间上每个集合最多 9 个元素，可视为 O(1)。

**代码**（`src/matrix/valid_sudoku.py` / `.cpp`）：

```python
def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val == ".":
                continue
            k = (i // 3) * 3 + j // 3
            if val in rows[i] or val in cols[j] or val in boxes[k]:
                return False
            rows[i].add(val)
            cols[j].add(val)
            boxes[k].add(val)
    return True
```

```cpp
bool isValidSudoku(const std::vector<std::string> &board) {
    std::vector<std::unordered_set<char>> rows(9), cols(9), boxes(9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            char val = board[i][j];
            if (val == '.') {
                continue;
            }
            int k = (i / 3) * 3 + j / 3;
            if (rows[i].count(val) || cols[j].count(val) || boxes[k].count(val)) {
                return false;
            }
            rows[i].insert(val);
            cols[j].insert(val);
            boxes[k].insert(val);
        }
    }
    return true;
}
```

- **复杂度**：时间 O(1)（固定 81 格，每格常数次集合操作），空间 O(1)（最多 81 个
  字符分布在 27 个集合里）。
- **易错点**：宫编号一定要用 `(i//3)*3 + j//3`，直接用 `i//3 + j//3` 或 `i*3+j`
  都错；空格的判断是字符 `'.'`，别写成整数 0；集合要先查后加，顺序反了会把自己
  判成冲突。
- **相似题**：79. 单词搜索（`docs/11-backtracking.md`，网格上的约束搜索）；
  37. 解数独（在本题的约束之上做回溯填数）；289. 生命游戏（同为网格模拟，
  见模式七）。

---

## 模式九：有序矩阵的阶梯查找

**适用信号**：矩阵的行、列各自有序（每行从左到右升序、每列从上到下升序），要查找
某值或做类似「排除」的操作。

**核心动作**：从**右上角**（或左下角）出发，一次比较就能排除一整行或一整列，最多
走 `m+n` 步。

### 240. 搜索二维矩阵 II（中等）

**题目**：给定 m x n 矩阵，每行从左到右升序、每列从上到下升序，判断目标值
`target` 是否存在。

**思路**：

从右上角 `(0, n-1)` 出发，反复比较：

- 等于 `target`，找到；
- 大于 `target`，说明这一列中它下方的数只会更大，不可能有 `target`，列下标左移；
- 小于 `target`，说明这一行中它左边的数只会更小，不可能有 `target`，行下标下移。

每比较一次就排除一整行或一整列，最多走 `m+n` 步。

**为什么必须从右上角（或左下角）出发**：右上角是这一行里最大的、同时是这一列里
最小的，所以一次比较就能确定「往左」还是「往下」。若从左上角出发，它是行、列的
最小值，两个方向的值都比它大，无法排除；右下角同理。

**代码**（`src/matrix/search_2d_matrix_ii.py` / `.cpp`）：

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    i, j = 0, len(matrix[0]) - 1
    while i < len(matrix) and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False
```

```cpp
bool searchMatrix(const std::vector<std::vector<int>> &matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) {
        return false;
    }
    int i = 0;
    int j = static_cast<int>(matrix[0].size()) - 1;
    while (i < static_cast<int>(matrix.size()) && j >= 0) {
        if (matrix[i][j] == target) {
            return true;
        } else if (matrix[i][j] > target) {
            --j;
        } else {
            ++i;
        }
    }
    return false;
}
```

- **复杂度**：时间 O(m+n)（每步排除一行或一列），空间 O(1)。
- **易错点**：起点必须是右上角或左下角，从左上/右下出发只能走 O(m*n)；两个边界
  条件是 `i < m` 且 `j >= 0`，缺一不可；空矩阵（`matrix` 为空或首行为空）要先判，
  否则 `matrix[0].size()` 会出错。
- **相似题**：74. 搜索二维矩阵（`docs/05-binary-search.md`，把二维展平后整体二分，
  条件是「按行首尾相接」也有序）；766. 托普利茨矩阵（本篇模式五，也利用行列的
  结构性质）；378. 有序矩阵中第 K 小的元素（`docs/08-heap.md`，多路归并取第 k 小）。

---

## 规律总结

1. **坐标变换先写公式，再抄代码**：转置是 `(i, j) → (j, i)`，旋转 90 度是
   `(i, j) → (j, n-1-i)`，重塑是经一维序号 `k` 换算。凡是「变形」题，先在纸上把
   旧位置与新位置的关系写清楚，代码就不会错。

2. **方阵才能原地转置**：非方阵转置会让行列数变化，只能新开矩阵；即便方阵，原地
   转置也要让内层循环从 `j = i + 1` 开始，避免同一对元素交换两次。

3. **旋转可拆成两个小动作**：顺时针 = 转置 + 水平翻转，逆时针 = 转置 + 垂直翻转。
   把复杂变换分解成「只交换两格」的步骤，比一次性写四元环交换更不容易写错。

4. **螺旋/边界遍历的通用骨架是四边界收缩**：`top/bottom/left/right` 各管一条边，
   每走完一条边就向内收一格；后半程的两条边要用 `if` 保护，否则奇数圈会重复覆盖。

5. **对角线用 `i + j = d` 定位**：`d` 从 0 到 `m+n-2`，`d` 的奇偶直接决定方向。
   把「二维遍历」换成「按一条条一维线遍历」，状态更少、边界更清晰。

6. **判断「线上是否一致」只比相邻两项**：托普利茨矩阵里每个元素和左上邻居比即可，
   靠相等/单调的传递性推广到整条线。能不两两比较就别两两比较。

7. **原地修改的核心是「别提前破坏还要用的数据」**：要么先把信息存进常数个变量
   （73 的 `col0`），要么借矩阵自己的行/列当备忘录（73 的首行首列），要么用整数的
   额外二进制位暂存新状态（289）。三条路都指向同一件事：找一个不会被后续读取的
   存储位置。

8. **当存储不够用时，复用输入本身**：73 与 289 是「用原地技巧把空间压到 O(1)」的
   两个范本。他们的额外空间都不来自新容器，而来自矩阵里本来就有、又不会被误读的
   位置或位。遇到「要求 O(1) 额外空间」的原地题，先问自己「输入里哪块空间是空闲的」。

9. **分组约束用「每组一个集合」校验**：数独的行、列、宫各是一类分组，各配一个
   集合，边扫边查重。分组编号常用整除与取模拼出来（`(i//3)*3 + j//3`），这类
   「按块编号」在分桶、分块统计里反复出现。

10. **行列有序的矩阵要从「一维极值交汇点」切**：右上角/左下角同时是某方向的最大、
    另一方向的最小，因此一次比较就能砍掉一整行或一整列，得到 O(m+n) 的阶梯查找。
    若矩阵只保证「行首尾相接整体有序」，则应展平后用二分（74 题）。

11. **二维题的边界判断是重灾区**：闭区间还是半开区间、从 0 还是从 1 开始、循环终止
    用 `<` 还是 `<=`——每个细节都可能让结果差一格。写完先拿 1x1、1xn、nx1、
    奇偶边长几组小样例手动过一遍，再谈优化。
