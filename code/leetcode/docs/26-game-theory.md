# 博弈论：必胜必败态与「只看分差」的区间 DP

博弈类题目乍看没有套路：没有现成的数据结构，题面还总在讲「两个人轮流、都最优」。但只要
抓住两条主线，绝大多数题都能归位：

1. **零和、完全信息、无随机**的博弈，只关心**分差**。两个人的总分之和是定值，谁赢只看
   差值，所以状态里存「当前行动者相对对手能领先多少」就够了，不必分别记录两人的分数。
2. **一个状态是「必胜」当且仅当存在一步能走到「必败」状态**。这条规则让我们可以从终局
   倒着推出每个状态的胜负，这就是「布尔胜负态 + 反向递推」。

除了这两条，还有一类题在「对手不知道我的数字 / 对手会挑最坏分支」的意义下是**对抗搜索**，
转移里要外层取 `min`（我选得最好）、内层取 `max`（对手把我推向最坏）；当状态很多且元素
可枚举时，再叠一层**状态压缩 + 记忆化**（用二进制位表示「哪些数被选过」）。

本篇收 10 道经典题，按五种套路分组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：找「平衡态」用数学定胜负 | 292. Nim 游戏 | 简单 |
| 模式二：布尔胜负态反向递推 | 1025. 除数博弈 / 1510. 石子游戏 IV | 简单 / 困难 |
| 模式三：分差区间 DP（轮流取两端） | 486. 预测赢家 / 877. 石子游戏 / 1690. 石子游戏 VII | 中等 / 中等 / 中等 |
| 模式四：带步长与上限的博弈 DP | 1140. 石子游戏 II / 1406. 石子游戏 III | 中等 / 困难 |
| 模式五：对抗搜索与状态压缩 | 375. 猜数字大小 II / 464. 我能赢吗 | 中等 / 中等 |

> 「分差」这个视角和第 13 篇「动态规划」里的状态设计是同一件事：好的状态定义能把二维的
> 「双方得分」压成一维的「差值」。而「必胜/必败」布尔态相当于把 DP 的取值从「数」换成
> 「真/假」，转移从「取 max/min」换成「存在一步」，本质仍是动态规划。

---

## 模式一：找「平衡态」用数学定胜负

**适用信号**：取石子的规则里有一个「固定的和」，且对手可以**镜像**你的操作。

**核心动作**：找出那个对手能始终维持、而你一旦面对就无法脱身的「必败局面」，它通常是
某个数的倍数。

### 292. Nim 游戏（简单）

**题目**：一堆 n 颗石子，两人轮流取 1~3 颗，取走最后一颗者胜。你先手，两人最优，问能否赢。

**思路**：

先手必败的局面恰好是 4 的倍数：

```python
def can_win_nim(n):
    return n % 4 != 0
```

```cpp
bool canWinNim(int n) {
    return n % 4 != 0;
}
```

**为什么是 4 的倍数**：若 n 是 4 的倍数，你取 1/2/3 颗，对手就取 3/2/1 颗，把总数凑成 4，
于是「还剩下 4 的倍数」这个局面又回到你手上。这样来回几轮，你最后会面对 0（无子可取）而
输。反过来，若 n 不是 4 的倍数，你先取 `n % 4` 颗，把 4 的倍数留给对手，你就成了刚才
「对手」的角色，必胜。**关键不是记住 4，而是看出「对手能凑出同一个和」**：如果每次可取
1~k 颗，胜负就由 n 是否是 `k + 1` 的倍数决定。

- **复杂度**：时间 O(1)，空间 O(1)。
- **易错点**：`n == 0` 按题目约束不会出现；写 `n % 4 == 0` 时别把 True/False 写反。
- **相似题**：1025. 除数博弈（这题看似复杂，其实也能看出「偶数必胜」的规律，见下一节）；
  这类「凑固定和」的思想与 04 篇「前缀和 / 同余」里凑倍数、凑余数是同一类观察。

---

## 模式二：布尔胜负态反向递推

**适用信号**：题面直接问「先手能否获胜」，没有分数，只有输赢。

**核心动作**：用 `win[i]`（或 `dp[i]`）表示「当前局面轮到我，我能否必胜」，转移写成

```text
win[i] = 存在一种走法，使 win[走完之后的局面] == False
```

即「只要有一招能把对手送进必败态，我就必胜」；一招都没有（或每一步对手都必胜）就是必败。

### 1025. 除数博弈（简单）

**题目**：初始数字 n。轮到的人选一个能整除当前 n 的 x（0 < x < n），把 n 换成 n - x；
无法操作者输。你先手，问能否获胜。

**思路**：

```python
def divisor_game(n):
    dp = [False] * (n + 1)
    for i in range(2, n + 1):
        for x in range(1, i):
            if i % x == 0 and not dp[i - x]:
                dp[i] = True
                break
    return dp[n]
```

```cpp
bool divisorGame(int n) {
    std::vector<bool> dp(n + 1, false);
    for (int i = 2; i <= n; ++i) {
        for (int x = 1; x < i; ++x) {
            if (i % x == 0 && !dp[i - x]) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}
```

**为什么这样定义胜负态**：`dp[i]` 就是上面那条规则——只要能找到一个因子 x，使得走完之后
对手面对 `dp[i-x] == False`，那我就能赢。边界 `dp[1] = False`：数字是 1 时没有任何合法
x，行动者直接无路可走而输。

**为什么答案其实是「n 为偶数」**：所有奇数 i 的因子都是奇数，`i - x` 必为偶数；而偶数可
取 `x = 1` 把局面变成奇数。于是偶数能一直把奇数丢给对手，奇数必败。代码里的 DP 是通用
套路，理解这个规律能让你 O(1) 秒答。

- **复杂度**：时间 O(n^2)，空间 O(n)。
- **易错点**：`x` 的范围是 `0 < x < n`，不能取 n；别忘了 x 要能整除 i；`dp[1]` 的初值
  是 False（必败），不要顺手写成 True。
- **相似题**：1510. 石子游戏 IV（把「合法操作」从因子换成完全平方数，骨架完全一样）。

### 1510. 石子游戏 IV（困难）

**题目**：一堆 n 颗石子，两人轮流取，每次必须取走一个完全平方数（1、4、9、…）颗，取走
最后一颗者胜。你先手，问能否获胜。

**思路**：

```python
def winner_square_game(n):
    win = [False] * (n + 1)
    for i in range(1, n + 1):
        s = 1
        while s * s <= i:
            if not win[i - s * s]:
                win[i] = True
                break
            s += 1
    return win[n]
```

```cpp
bool winnerSquareGame(int n) {
    std::vector<bool> win(n + 1, false);
    for (int i = 1; i <= n; ++i) {
        for (int s = 1; s * s <= i; ++s) {
            if (!win[i - s * s]) {
                win[i] = true;
                break;
            }
        }
    }
    return win[n];
}
```

**为什么与 1025 是同一道题**：把「合法操作集合」抽象出来，两题都是「枚举我所有能走的
一步，只要有一招能让对手必败，我就必胜」。1025 的合法操作是「减去一个真因子」，1510 的
合法操作是「减去一个平方数」，转移的骨架一字不差。`win[0] = False`（没石子可取就输）是
两题共同的终局。

- **复杂度**：时间 O(n√n)，空间 O(n)。
- **易错点**：平方数从 1 开始枚举，循环条件是 `s*s <= i`；`win[0]` 必须是 False，它是
  整条递推的「地基」。
- **相似题**：1025. 除数博弈（同上）；这一类「枚举走法 + 布尔胜负态」也是 11 篇回溯里
  「决策树」的博弈版本。

---

## 模式三：分差区间 DP（轮流取两端）

**适用信号**：一排数，两人轮流从**两端**取，比总分大小。

**核心动作**：令 `dp[i][j]` = 在区间 `[i, j]` 上，轮到行动的人能取得的**最大分差**
（自己 − 对手）。取左端得到 `nums[i]`，但对手随后会成为 `[i+1, j]` 上的行动者、领先
`dp[i+1][j]`，于是净分差为 `nums[i] - dp[i+1][j]`；取右端同理。转移取较大者。三题只差
「得分怎么算」。

### 486. 预测赢家（中等）

**题目**：数组 nums，两人轮流从当前两端取一个数计入总分，都最优。问先手总分能否 ≥ 后手。

**思路**：

```python
def predict_the_winner(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = nums[i]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0
```

```cpp
bool predictTheWinner(std::vector<int> nums) {
    int n = nums.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        dp[i][i] = nums[i];
    }
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            dp[i][j] = std::max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1] >= 0;
}
```

**为什么用分差而不是双方得分**：总分固定为 `sum(nums)`，谁领先只看差值。把「双方得分」
压成一个差分，状态少一维、转移也更自然：我这一手拿走的 `nums[i]` 会**翻转**对手原本能
领先的分差，所以是减去 `dp[i+1][j]`。单元素区间 `dp[i][i] = nums[i]`，因为取走它后分差
就是它本身。最后 `dp[0][n-1] >= 0` 表示先手不败（题目规定平局也算先手赢）。

- **复杂度**：时间 O(n^2)，空间 O(n^2)（可用滚动数组降到 O(n)）。
- **易错点**：区间长度必须**从小到大**枚举，且 `dp[i][j]` 依赖更短的区间；返回是 `>= 0`
  而非 `> 0`；`n == 1` 时直接返回 True。
- **相似题**：877. 石子游戏（同一递推，问法稍不同）；1690. 石子游戏 VII（同一取法，
  但「得分是剩下石子的和」）。

### 877. 石子游戏（中等）

**题目**：偶数堆石子排成一行，两人轮流从两端取走一整堆计入总分，总分多者胜。问先手是否
获胜（题目保证堆数为偶数）。

**思路**：

```python
def stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = piles[i]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1])
    return dp[0][n - 1] >= 0
```

```cpp
bool stoneGame(std::vector<int> piles) {
    int n = piles.size();
    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        dp[i][i] = piles[i];
    }
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            dp[i][j] = std::max(piles[i] - dp[i + 1][j], piles[j] - dp[i][j - 1]);
        }
    }
    return dp[0][n - 1] >= 0;
}
```

**为什么和 486 一模一样**：两道题只差一个说法——486 问「先手总分能否 ≥ 后手」，877 问
「先手是否获胜」，而得分规则都是「取走的两端元素计入自己」。把 877 的约束（偶数堆）用
上，甚至能证明先手可以把石子按奇偶下标分成两组、始终吃较大的一组，从而必胜（`return
True` 也能过）。这里仍用 DP，是为了给出不看穿结论时也能做的方法。

- **复杂度**：时间 O(n^2)，空间 O(n^2)。
- **易错点**：这题堆数必为偶数，所以理论上先手不败；但别把 DP 里的「分差」误当成某一方
  总分；返回是 `>= 0`。
- **相似题**：486. 预测赢家（同型，交叉对照）；1690. 石子游戏 VII（下面马上讲，
  取法相同但得分规则不同）。

### 1690. 石子游戏 VII（中等）

**题目**：一排石子 stones，两人轮流从两端拿走一堆。这一手的得分是「拿完之后**剩下**的
石子总数」（拿走的那堆不计分）。求 Alice 相对 Bob 的最大分差。

**思路**：

```python
def stone_game_vii(stones):
    n = len(stones)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]

    def range_sum(i, j):
        return prefix[j + 1] - prefix[i]

    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            left = range_sum(i + 1, j) - dp[i + 1][j]
            right = range_sum(i, j - 1) - dp[i][j - 1]
            dp[i][j] = max(left, right)
    return dp[0][n - 1]
```

```cpp
int stoneGameVII(std::vector<int> stones) {
    int n = stones.size();
    std::vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + stones[i];
    }
    auto rangeSum = [&](int i, int j) { return prefix[j + 1] - prefix[i]; };

    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            int left = rangeSum(i + 1, j) - dp[i + 1][j];
            int right = rangeSum(i, j - 1) - dp[i][j - 1];
            dp[i][j] = std::max(left, right);
        }
    }
    return dp[0][n - 1];
}
```

**为什么 base case 是 0**：这一手取左端时，得分是「剩下的和」`sum(i+1..j)`，而不是取走
的元素值。所以当区间只剩一个元素时，拿走它之后剩余为空、得分 0，`dp[i][i] = 0`。这正
是和 486 / 877 的关键区别：取法一样，但「得分」的口径变了，base case 就跟着变。区间和
用前缀和 O(1) 求出，转移仍然是「本次得分 − 对手在剩余区间能领先的分差」。

- **复杂度**：时间 O(n^2)，空间 O(n^2)（可滚动到 O(n)）。
- **易错点**：别把「得分 = 剩下和」误写成「得分 = 取走的值」；`dp[i][i] = 0`；区间和函数
  的下标是闭区间 `[i, j]`，前缀和里对应 `prefix[j+1] - prefix[i]`。
- **相似题**：486. 预测赢家、877. 石子游戏（三者取法相同，仅得分口径不同，务必横向对比）。

---

## 模式四：带步长与上限的博弈 DP

**适用信号**：取石子不是「两端各取一个」，而是「从头连续取若干堆」，且能取多少受某个
参数限制（比如 M）。

**核心动作**：把「位置」和「限制参数」一起放进状态，转移仍是「我拿到的 + 剩余归对手」。
如果题目要的是**总量**而不是分差，就利用「剩余总量 = 我拿的 + 对手拿的」把对手的部分
减出来。

### 1140. 石子游戏 II（中等）

**题目**：石子排成一行 piles。Alice 先手，每次从队首连续取 X 堆（1 ≤ X ≤ 2M，初始
M = 1），取完把 M 更新为 max(M, X)。两人最优，问先手最多能拿多少石子。

**思路**：

```python
from functools import lru_cache


def stone_game_ii(piles):
    n = len(piles)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + piles[i]

    @lru_cache(maxsize=None)
    def f(i, m):
        if i >= n:
            return 0
        best = 0
        for x in range(1, 2 * m + 1):
            if i + x > n:
                break
            best = max(best, suffix[i] - f(i + x, max(m, x)))
        return best

    return f(0, 1)
```

```cpp
int stoneGameII(std::vector<int> piles) {
    int n = piles.size();
    std::vector<int> suffix(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        suffix[i] = suffix[i + 1] + piles[i];
    }
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, -1));
    std::function<int(int, int)> f = [&](int i, int m) -> int {
        if (i >= n) {
            return 0;
        }
        if (dp[i][m] != -1) {
            return dp[i][m];
        }
        int best = 0;
        for (int x = 1; x <= 2 * m; ++x) {
            if (i + x > n) {
                break;
            }
            best = std::max(best, suffix[i] - f(i + x, std::max(m, x)));
        }
        return dp[i][m] = best;
    };
    return f(0, 1);
}
```

**为什么状态里要有 M**：能不能取第 i+1 堆、一次最多能取几堆，取决于当前的 M，而 M 会
随着「上一次取了多少」变化。所以「位置 i」之外必须再记一个「上限 M」。设 `f(i, m)` 是
「从第 i 堆开始、上限为 m 时，行动者最终能拿到的**总石子数**」。我取前 x 堆拿到
`suffix[i] - suffix[i+x]`，剩下的石子由两人分完，对手在 `i+x` 处能拿 `f(i+x, max(m,x))`，
所以「我剩下的那部分」= 剩余总量 − 对手能拿的，于是

```text
f(i, m) = max over 1 <= x <= 2m  of  ( suffix[i] - f(i + x, max(m, x)) )
```

**为什么 M 单调不减**：M 只会在 `max(m, x)` 里增大，意味着越往后一手能取得越多。这不是
bug，而是题目的规则，也正是状态要说清 M 的原因。

- **复杂度**：时间 O(n^3)，空间 O(n^2)（n ≤ 100）。
- **易错点**：`x` 的上界是 `2*m`；`i + x > n` 就停；状态要记 `(i, m)` 而不是只记 i；
  C++ 记忆化数组里 m 的维度开 `n+1` 即可（`max(m, x)` 不会超过 n）。
- **相似题**：1406. 石子游戏 III（把「上限 M」换成固定步长 1/2/3）；877. 石子游戏
  （若把步长限制去掉，就退化成取两端的博弈）。

### 1406. 石子游戏 III（困难）

**题目**：石子排成一行 stoneValue。两人轮流从队首取 1、2 或 3 堆计入总分，取完为止。
两人最优，问先手总分能否**严格高于**后手。

**思路**：

```python
def stone_game_iii(stone_value):
    n = len(stone_value)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + stone_value[i]

    dp = [0] * (n + 4)
    for i in range(n - 1, -1, -1):
        best = float("-inf")
        for x in (1, 2, 3):
            if i + x <= n:
                take = suffix[i] - suffix[i + x]
                best = max(best, take - dp[i + x])
        dp[i] = best
    return dp[0] > 0
```

```cpp
bool stoneGameIII(std::vector<int> stoneValue) {
    int n = stoneValue.size();
    std::vector<int> suffix(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        suffix[i] = suffix[i + 1] + stoneValue[i];
    }
    std::vector<int> dp(n + 4, 0);
    for (int i = n - 1; i >= 0; --i) {
        int best = INT_MIN;
        for (int x = 1; x <= 3; ++x) {
            if (i + x <= n) {
                int take = suffix[i] - suffix[i + x];
                best = std::max(best, take - dp[i + x]);
            }
        }
        dp[i] = best;
    }
    return dp[0] > 0;
}
```

**为什么这题用分差、1140 用总量**：两题其实可以互相改写，但各自最自然的形式不同。1140
问「先手拿多少（总量）」，而 1406 问「先手是否赢（比分）」，所以这里用最常见的分差定义：
`dp[i]` = 从第 i 堆开始行动者能领先对手的分数。取前 x 堆当场得 `suffix[i]-suffix[i+x]`，
再减去对手在 `i+x` 处能领先的 `dp[i+x]`。只依赖 `dp[i+1..i+3]`，所以从右往左一遍即可，
不需要区间 DP。`dp[n] = 0`（没石子时无分差），最后 `dp[0] > 0` 判定严格获胜。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：只取 1/2/3 堆，别写成 `1..n`；`i + x <= n` 的边界要判；返回严格大于 0，
  平局（`dp[0] == 0`）不算先手赢。
- **相似题**：1140. 石子游戏 II（带 M 的版本）；486. 预测赢家（分差 DP 的另一形态，
  只是取法从「取一堆」变成「取一段」）。

---

## 模式五：对抗搜索与状态压缩

**适用信号**：转移里外层「我选最优」、内层「对手选最坏」，是典型的 minimax；或者可选
元素很多、需要一个能当哈希键的状态表示。

**核心动作**：对抗博弈写成 `dp[状态] = min/max over 我的动作 of (代价 + max/min over
对手分支)`；当「可选集合」可枚举时，用一个整数的二进制位表示「谁还没被选」，配上记忆化
搜索。

### 375. 猜数字大小 II（中等）

**题目**：心里想一个 1~n 的数。每次猜 x，猜错会被告知「大了/小了」，并支付 x 元。无论
答案是几都能保证猜中，问最少要准备多少钱（最坏情况花费）。

**思路**：

```python
def get_money_amount(n):
    dp = [[0] * (n + 2) for _ in range(n + 2)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            best = float("inf")
            for x in range(i, j + 1):
                cost = x + max(dp[i][x - 1], dp[x + 1][j])
                if cost < best:
                    best = cost
            dp[i][j] = best
    return dp[1][n] if n >= 1 else 0
```

```cpp
int getMoneyAmount(int n) {
    std::vector<std::vector<int>> dp(n + 2, std::vector<int>(n + 2, 0));
    for (int len = 2; len <= n; ++len) {
        for (int i = 1; i + len - 1 <= n; ++i) {
            int j = i + len - 1;
            int best = INT_MAX;
            for (int x = i; x <= j; ++x) {
                int cost = x + std::max(dp[i][x - 1], dp[x + 1][j]);
                best = std::min(best, cost);
            }
            dp[i][j] = best;
        }
    }
    return dp[1][n];
}
```

**为什么内层取 max、外层取 min**：猜 x 之后，答案要么落在左边 `[i, x-1]`、要么落在右边
`[x+1, j]`。对手（出题人）会把情况引向你更亏的那一边，所以最坏花费是两边的**较大值**；
而你要在 x 的所有选择里挑让最坏花费最小的，所以外层取 **min**。即

```text
dp[i][j] = min over x in [i, j] of ( x + max(dp[i][x-1], dp[x+1][j]) )
```

这是「对抗搜索」最标准的形状：`minmax`。区间长度为 1 时一眼猜中、不花钱，`dp[i][i] = 0`。

- **复杂度**：时间 O(n^3)，空间 O(n^2)（n ≤ 200）。
- **易错点**：这不是「二分查找」——因为每次猜错都要付钱、且对手总选更坏的分支；转移
  里 `max` 与 `min` 的层次不能颠倒；边界空区间记 0。
- **相似题**：464. 我能赢吗（同为对抗搜索，但用状态压缩 + 记忆化）；23 篇「二分」里
  的二分答案是「每步代价相同」的简化版，本题因代价不对称而不能直接二分。

### 464. 我能赢吗（中等）

**题目**：从 1 到 maxChoosableInteger 里，两人轮流选一个没被选过的数累加。谁先让累计和
达到（或超过）desiredTotal 谁获胜。你先手，问能否保证获胜。

**思路**：

```python
def can_i_win(max_choosable_integer, desired_total):
    if desired_total <= 0:
        return True
    total = max_choosable_integer * (max_choosable_integer + 1) // 2
    if total < desired_total:
        return False
    memo = {}

    def win(mask, total):
        if mask in memo:
            return memo[mask]
        for i in range(1, max_choosable_integer + 1):
            bit = 1 << (i - 1)
            if mask & bit:
                continue
            if total + i >= desired_total or not win(mask | bit, total + i):
                memo[mask] = True
                return True
        memo[mask] = False
        return False

    return win(0, 0)
```

```cpp
bool canIWin(int maxChoosableInteger, int desiredTotal) {
    if (desiredTotal <= 0) {
        return true;
    }
    long long total = 1LL * maxChoosableInteger * (maxChoosableInteger + 1) / 2;
    if (total < desiredTotal) {
        return false;
    }
    std::unordered_map<int, bool> memo;
    std::function<bool(int, int)> win = [&](int mask, int sum) -> bool {
        auto it = memo.find(mask);
        if (it != memo.end()) {
            return it->second;
        }
        for (int i = 1; i <= maxChoosableInteger; ++i) {
            int bit = 1 << (i - 1);
            if (mask & bit) {
                continue;
            }
            if (sum + i >= desiredTotal || !win(mask | bit, sum + i)) {
                memo[mask] = true;
                return true;
            }
        }
        memo[mask] = false;
        return false;
    };
    return win(0, 0);
}
```

**为什么用位掩码**：唯一的变量是「哪些数被选过」，而 maxChoosableInteger ≤ 20，正好能用
一个整数的低 20 位表示：第 i 位为 1 表示数字 i 已用。为什么记忆化的键只用 `mask`、不用
再把累计和也放进去？因为累计和完全由「选了哪些数」决定，是 `mask` 的函数，不会产生歧义。
递归规则还是那句老话：只要存在一个没选过的 i，使「加上它立刻达标」或「对手在新状态下
必败」，当前玩家就必胜。

**两个边界**：`desiredTotal <= 0` 时先手直接赢；若所有数之和都小于目标，谁也达不了标，
按题目判定先手输，提前返回 False。

- **复杂度**：时间 O(2^n · n)，空间 O(2^n)，n = maxChoosableInteger（n ≤ 20）。
- **易错点**：位掩码的下标偏移（数字 i 对应第 `i-1` 位）；记忆化键只需 mask；上界
  `1<<(i-1)` 在 n=20 时是 1<<19，仍适合 32 位 int。
- **相似题**：375. 猜数字大小 II（同为对抗搜索，一个用区间状态、一个用位掩码状态）；
  16 篇「位运算」的位枚举（用整数的每一位当开关表），本页把它用在了状态压缩上。

---

## 规律总结

1. **先分清题目问的是「输赢」还是「分数」**。问输赢 → 布尔胜负态 `win[i]`，转移是「存在
   一步走到对手必败」（1025、1510）；问分差 → 记「当前行动者能领先多少」，转移里减去
   对手的领先值（486、877、1406、1690）。两类都可能用同一套「枚举走法」，只是取值类型
   不同。

2. **零和博弈只关心分差**，不用分别存两个人的总分。`dp[i][j] = max(取左 − dp[i+1][j],
   取右 − dp[i][j-1])` 这一个式子覆盖了 486 / 877；得分口径一变（1690 的「剩下和」），
   只需换掉「本次得分」和 base case。

3. **「对手能镜像/凑数」时，答案常常是一个整除判断**。292 的模 4、1025 的奇偶，都是先
   在草稿纸上从小数据找规律、再回头证明。看到「每次可取 1..k」就试着想 `k+1` 的倍数。

4. **状态要放全决定下一步的量**。1140 里「最多能取几堆」由 M 决定，所以状态必须记
   `(i, M)`；漏掉 M 会算错。判断状态是否完整，就问一句：**给定这个状态，下一步的所有
   选择是否都确定了？**

5. **minimax 的取极值层次别写反**：外层是「我选最优」（min 或 max 由问法决定），内层是
   「对手把我推向最坏」。375 的 `x + max(dp左, dp右)` 再对 x 取 min 就是标准形状。

6. **可枚举集合就压成位**。当状态是「哪些元素被选过」且数量不大（≤ 20）时，用一个整数
   的二进制位表示，直接当记忆化的键。别忘了「累计和是 mask 的函数」，所以键只需 mask。

7. **反向递推时先想「终局」**。布尔态的 `dp[0] = False`（1025、1510）、区间 DP 的
   `dp[i][i]`（486 是元素值、1690 是 0）、`dp[n] = 0`（1406）——终局的定义直接决定整条
   递推对不对。写下转移前，先把终局老老实实写出来。

8. **区间 DP 的遍历顺序**：`dp[i][j]` 依赖更短的区间，所以按**区间长度从小到大**枚举；
   只依赖右侧元素的后缀型递推（1406）则可以简单地**从右往左**扫。选对顺序就不用手写
   记忆化。

9. **与其它篇的联系**：分差 DP 和 13 篇「动态规划」的状态设计同源；位掩码状态与 16 篇
   「位运算」的位枚举同源；1140 的「剩余总量 = 我拿的 + 对手拿的」和 04 篇前缀和里
   「区间和 = 两前缀之差」是同一套守恒关系。
