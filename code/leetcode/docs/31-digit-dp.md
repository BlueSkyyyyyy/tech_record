# 数位 DP：按位枚举、掩码与记忆化

有一类计数问题长这样：给定一个上界 N，问 `[0, N]`（或 `[L, R]`）里有多少个数
满足某种条件。直接一个个数太慢，但数的十进制写法是从高位到低位展开的一棵「决策
树」——每一位填 0..9，填完 L 位就得到一个数。数位 DP 就是在这棵树上**逐位填、
把重复出现的状态记下来**。

真正让这棵树「收得住」的，是三个开关：

- **`tight`（贴不贴上界）**：前面的位都填得和 N 一模一样时，当前位不能超过 N 的
  对应位；一旦某一位填小了，后面就自由了。贴着上界的分支只有一条，不影响复杂度。
- **`started`（是否已开始）**：用来跳过前导零。比如 N 有 5 位，数 7 其实是
  `00007`，前导的那几个 0 不算「用掉了数字 0」，也不能当成一位去比较。题意涉及
  正数时，它还能顺手把 0 排除掉。
- **位置对应的状态**：数位 DP 的精华全在这里——条件不同，状态就不同。要不要记
  「上一位填了几」（相邻位约束）、记「用过的数字掩码」（各位互不相同）、记
  「数位和」（和落在某段区间），都由题目条件决定。

区间 `[L, R]` 用**前缀相减** `f(R) - f(L-1)` 化归成 `[0, X]`；当 L、R 是大数字
符串时，「减一」就直接对字符串做。

本篇 10 题按「状态记什么」分四组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：数位出现次数（按位乘法原理） | 233. 数字 1 的个数 / 1067. 范围内的数字计数 | 困难 / 困难 |
| 模式二：各位互不相同的计数 | 357. 统计各位数字都不同的数字个数 / 1012. 至少有 1 位重复的数字 / 2376. 统计特殊整数 | 中等 / 困难 / 困难 |
| 模式三：相邻位的约束 | 600. 不含连续 1 的非负整数 / 1215. 步进数 / 2801. 统计范围内的步进数字数目 | 困难 / 中等 / 困难 |
| 模式四：固定数字集与数位和 | 902. 最大为 N 的数字组合 / 2719. 统计整数数目 | 困难 / 困难 |

> 一句话记住数位 DP 的骨架：**固定上界的每一位 → 用 `tight` 控制是否受限 →
> 用 `started` 处理前导零 → 把「已确定的信息」定义成状态 → 记忆化**。

---

## 模式一：数位出现次数（按位乘法原理）

**适用信号**：要统计某个数字在所有数里一共出现多少次。

**核心动作**：总次数 = 每一位上出现该数字的次数之和。固定权重 `p = 10^k`，把
`0..N` 按这一位切成 `high | cur | low` 三段，用乘法原理分 `cur` 与目标数字的
大小关系讨论。这一组的状态可以完全闭式算出，连记忆化都不需要。

### 233. 数字 1 的个数（困难）

**题目**：给定整数 n，统计所有小于等于 n 的非负整数中数字 1 出现的总次数。

**思路**：

```python
def count_digit_one(n):
    count = 0
    p = 1
    while p <= n:
        high = n // (p * 10)
        cur = (n // p) % 10
        low = n % p
        if cur > 1:
            count += (high + 1) * p
        elif cur == 1:
            count += high * p + low + 1
        else:
            count += high * p
        p *= 10
    return count
```

```cpp
long long countDigitOne(long long n) {
    long long count = 0;
    for (long long p = 1; p <= n; p *= 10) {
        long long high = n / (p * 10);
        int cur = static_cast<int>((n / p) % 10);
        long long low = n % p;
        if (cur > 1) {
            count += (high + 1) * p;
        } else if (cur == 1) {
            count += high * p + low + 1;
        } else {
            count += high * p;
        }
    }
    return count;
}
```

**为什么这样分类**：固定一个位（比如十位），把它取 1 时能配出多少个数，只取决于
这一位原来的数字 `cur`。以 n = 213 的十位为例：`high = 2, cur = 1, low = 3`。

- 若 `cur > 1`：这一位取 1 后高位可取 0..high 共 high+1 种，低位 0..p-1 共 p 种；
- 若 `cur == 1`：高位比 high 小的时候低位随便（high * p 种）；高位正好等于 high
  时，低位不能超过 `low`（low+1 种）；
- 若 `cur < 1`（即 0）：高位只能取到 high-1，低位随便。

三类互不重叠、合起来正好覆盖所有数，逐位相加即答案。

- **复杂度**：时间 O(log n)（数的位数），空间 O(1)。
- **易错点**：`p` 要用 `long long`，`p * 10` 别溢出；循环条件是 `p <= n`，
  n = 0 时直接返回 0。这是「按位统计」而不是「枚举数字」，如果去构造每个数会超时。
- **相似题**：1067（下题）把目标数字换成任意 d；面试题 17.06「2 出现的次数」是
  d = 2 的特例；357（模式二）则是把「按位」思想用在另一种条件上。

### 1067. 范围内的数字计数（困难）

**题目**：给定整数 d（0..9）、low、high，统计数字 d 在区间 `[low, high]` 的所有
整数中出现的总次数。

**思路**：

```python
def _count_upto(n, d):
    if n <= 0:
        return 0
    count = 0
    p = 1
    while p <= n:
        high = n // (p * 10)
        cur = (n // p) % 10
        low = n % p
        if d != 0:
            if cur > d:
                count += (high + 1) * p
            elif cur == d:
                count += high * p + low + 1
            else:
                count += high * p
        else:
            if high > 0:
                if cur == 0:
                    count += (high - 1) * p + low + 1
                else:
                    count += high * p
        p *= 10
    return count


def digit_count_in_range(d, low, high):
    return _count_upto(high, d) - _count_upto(low - 1, d)
```

```cpp
long long countUpto(long long n, int d) {
    if (n <= 0) {
        return 0;
    }
    long long count = 0;
    for (long long p = 1; p <= n; p *= 10) {
        long long high = n / (p * 10);
        int cur = static_cast<int>((n / p) % 10);
        long long low = n % p;
        if (d != 0) {
            if (cur > d) {
                count += (high + 1) * p;
            } else if (cur == d) {
                count += high * p + low + 1;
            } else {
                count += high * p;
            }
        } else if (high > 0) {
            if (cur == 0) {
                count += (high - 1) * p + low + 1;
            } else {
                count += high * p;
            }
        }
    }
    return count;
}

int digitCountInRange(int d, int low, int high) {
    return static_cast<int>(countUpto(high, d) - countUpto(low - 1, d));
}
```

**为什么区间用相减**：这正是「前缀和」的思想，只不过这里的前缀量是「数字 d 出现
的次数」。`[low, high]` 的次数 = `0..high` 的次数减 `0..low-1` 的次数。函数
`_count_upto` 的框架和 233 完全一样，唯一的区别是目标数字不一定为 1。

**d = 0 为什么要多一层判断**：0 不能当「前导零」混进来。比如数 7 写成 `007`，
那两个 0 不算出现。所以 `high == 0`（这一位是最前面的有效位之前）时要跳过；并且
`cur == 0` 时，高位不能从 0 起算（否则「这一位是 0」的情况会和更短的数字重复计
数），要从 1 起，即高位有 `high - 1` 种。

- **复杂度**：时间 O(log high)，空间 O(1)。
- **易错点**：`d == 0` 的两种情况最容易写错；`low = 0` 时 `_count_upto(-1, d)`
  要能返回 0（代码里用 `n <= 0` 挡住）；`high - 1` 处 high 已保证大于 0。
- **相似题**：233（上题）是 d = 1 且 low = 0 的特例；面试题 17.06 是 d = 2。

---

## 模式二：各位互不相同的计数

**适用信号**：条件取决于「某个数字在这之前有没有用过」，典型是「各位数字互不相同」。

**核心动作**：用一个 10 位二进制掩码 `mask` 记录已经用过的数字，填新数字时先用
`mask & (1 << d)` 判重，再 `mask | (1 << d)` 并入。`started` 保证前导零不算用掉
数字 0。这一组还有一个分支：当上界恰好是 10 的幂时，答案有闭式，可以用乘法原理
直接算，不必记忆化——357 就是。

### 357. 统计各位数字都不同的数字个数（中等）

**题目**：给定 n，统计 `[0, 10^n)` 内各位数字都不同的整数个数。

**思路**：

```python
def count_numbers_with_unique_digits(n):
    if n == 0:
        return 1
    total = 10
    cur = 9
    for length in range(2, n + 1):
        cur *= 10 - length + 1
        total += cur
    return total
```

```cpp
int countNumbersWithUniqueDigits(int n) {
    if (n == 0) {
        return 1;
    }
    long long total = 10;
    long long cur = 9;
    for (int length = 2; length <= n; ++length) {
        cur *= 10 - length + 1;
        total += cur;
    }
    return static_cast<int>(total);
}
```

**为什么能闭式算**：n 给的是「位数上限」，上界正好是 `10^n`，也就是位数不超过 n
的所有数，于是可以按长度分段用乘法原理：

- 长度 0：只有 0，1 个；长度 1：0..9，10 个；
- 长度 L ≥ 2：首位 9 种（非 0），第二位可选剩下的 9 个（含 0），第三位剩 8 个……
  第 L 位剩 `10 - L + 1` 个。

所以长度 L 的个数是 `9 * 9 * 8 * ...`，累加长度 1..n 即可。这和数位 DP 里的掩码
状态是一回事——只是当上界是整幂时，「还剩几个数字可用」这个信息就足够，掩码被压
缩成了一个数字。

- **复杂度**：时间 O(min(n, 10))，空间 O(1)。
- **易错点**：n = 0 返回 1（只有 0 自己）；长度超过 10 后乘积自然为 0（鸽巢原理），
  不用特判，但循环到 n = 11 时那一项确实是 0。C++ 里 `total`、`cur` 用 `long long`
  防中间量溢出。
- **相似题**：2376（下下题）把「位数上限」换成具体上界，只能记忆化；1012（下题）
  用「不同数的补集」求「至少一位重复」。

### 1012. 至少有 1 位重复的数字（困难）

**题目**：给定正整数 n，返回 `[1, n]` 内「至少有一位数字重复」的整数个数。

**思路**：

```python
from functools import lru_cache


def _count_unique_upto(n):
    s = str(n)
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, mask, tight, started):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and d == limit
            if not started and d == 0:
                total += dfs(pos + 1, mask, ntight, False)
            elif mask & (1 << d):
                continue
            else:
                total += dfs(pos + 1, mask | (1 << d), ntight, True)
        return total

    return dfs(0, 0, True, False)


def num_dup_digits_at_most_n(n):
    return n - (_count_unique_upto(n) - 1)
```

```cpp
std::string s;
int length;
long long memo[12][1 << 10][2][2];

long long dfs(int pos, int mask, bool tight, bool started) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][mask][tight][started];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, mask, ntight, false);
        } else if (mask & (1 << d)) {
            continue;
        } else {
            total += dfs(pos + 1, mask | (1 << d), ntight, true);
        }
    }
    return res = total;
}

long long countUniqueUpto(long long n) {
    s = std::to_string(n);
    length = static_cast<int>(s.size());
    std::memset(memo, -1, sizeof(memo));
    return dfs(0, 0, true, false);
}

int numDupDigitsAtMostN(int n) {
    return static_cast<int>(n - (countUniqueUpto(n) - 1));
}
```

**为什么反过来数**：直接数「有重复」要在状态里记「是否已经重复过」，而且一旦重复
就没法用掩码判重了；反过来，`[1, n]` 一共有 n 个数，减去「各位互不相同」的个数就
是答案。于是问题化成数「不同数」，这正是掩码能胜任的。

`_count_unique_upto(n)` 统计 `[0, n]` 的不同数：沿 n 的十进制串逐位填，`mask` 记
用过的数字。注意 `not started and d == 0` 时掩码不变——前导零不占用数字 0；填其它
数字时若已在掩码中则这条分支作废。走到串尾返回 1，即构成一个合法数（含 0）。
最后 `[1, n]` 的不同数要减掉 0，所以是 `_count_unique_upto(n) - 1`。

- **复杂度**：时间 O(位数 × 2^10 × 2 × 2)，空间同阶；位数不超过 10。
- **易错点**：前导零必须用 `started` 单独处理，否则 0 会被判成重复；返回值用
  `long long`；`n - (...)` 里的括号别写漏。
- **相似题**：2376（下题）问的几乎就是「不同数」本身；357（上题）是它的整幂闭式版。

### 2376. 统计特殊整数（困难）

**题目**：如果一个正整数的每一位数字都互不相同，就称它是「特殊整数」。给定 n，
返回 `[1, n]` 内特殊整数的个数。

**思路**：

```python
from functools import lru_cache


def count_special_integers(n):
    s = str(n)
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, mask, tight, started):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and d == limit
            if not started and d == 0:
                total += dfs(pos + 1, mask, ntight, False)
            elif mask & (1 << d):
                continue
            else:
                total += dfs(pos + 1, mask | (1 << d), ntight, True)
        return total

    return dfs(0, 0, True, False) - 1
```

```cpp
std::string s;
int length;
long long memo[12][1 << 10][2][2];

long long dfs(int pos, int mask, bool tight, bool started) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][mask][tight][started];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, mask, ntight, false);
        } else if (mask & (1 << d)) {
            continue;
        } else {
            total += dfs(pos + 1, mask | (1 << d), ntight, true);
        }
    }
    return res = total;
}

int countSpecialIntegers(int n) {
    s = std::to_string(n);
    length = static_cast<int>(s.size());
    std::memset(memo, -1, sizeof(memo));
    return static_cast<int>(dfs(0, 0, true, false) - 1);
}
```

**和 1012 差在哪**：DP 部分一字不差——特殊整数就是「各位互不相同」的正整数。
区别只在最后一步：1012 要的是「有重复」，所以答案是 `n - (不同数 - 1)`；而本题
直接要「不同数」，答案就是 `dfs(...) - 1`（减掉 0）。这也说明数位 DP 的模板一旦
写好，同一段代码换个收尾就是另一道题。

- **复杂度**：时间 O(位数 × 2^10 × 2 × 2)，空间同阶。
- **易错点**：`- 1` 是减掉 0，别漏；`started` 的前导零分支必须保留，否则会把
  `000` 当成一个用掉了 0 的合法数。
- **相似题**：1012（上题）、357（上上题）是同一主题的三种问法和三种解法。

---

## 模式三：相邻位的约束

**适用信号**：条件只跟「相邻两位」的关系有关，比如不能出现连续 1、或者相邻数字
必须相差 1。

**核心动作**：状态里只记「上一位填了什么」（或「上一位是不是 1」），填下一位时
当场判断合法性。当上界是二进制时（600），逐位填 0/1 即可。

### 600. 不含连续 1 的非负整数（困难）

**题目**：给定正整数 n，返回 `[0, n]` 内二进制表示不含连续两个 1 的整数个数。

**思路**：

```python
from functools import lru_cache


def find_integers(n):
    s = bin(n)[2:]
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, prev_one, tight):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 1
        total = 0
        for b in range(limit + 1):
            if prev_one and b == 1:
                continue
            total += dfs(pos + 1, b == 1, tight and b == limit)
        return total

    return dfs(0, False, True)
```

```cpp
std::string bits;
int length;
long long memo[64][2][2];

long long dfs(int pos, bool prevOne, bool tight) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][prevOne][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? bits[pos] - '0' : 1;
    long long total = 0;
    for (int b = 0; b <= limit; ++b) {
        if (prevOne && b == 1) {
            continue;
        }
        total += dfs(pos + 1, b == 1, tight && b == limit);
    }
    return res = total;
}

int findIntegers(int n) {
    bits.clear();
    int x = n;
    if (x == 0) {
        bits = "0";
    }
    while (x > 0) {
        bits = static_cast<char>('0' + x % 2) + bits;
        x /= 2;
    }
    length = static_cast<int>(bits.size());
    std::memset(memo, -1, sizeof(memo));
    return static_cast<int>(dfs(0, false, true));
}
```

**为什么状态这么少**：能不能在这一位填 1，只看上一位是不是 1——上一位是 1，这一位
就只能填 0；上一位是 0，这一位 0/1 都行。所以状态只需 `prev_one` 一个布尔量。

**为什么不用处理前导零**：二进制下前导零不影响「有没有连续 1」，全 0 也天然合法，
所以从最高位老老实实填 0/1 就行，不需要 `started`。走到串尾返回 1，表示这个 0/1
串对应一个合法数（包括全 0）。

- **复杂度**：时间 O(位数)，空间 O(位数)。
- **易错点**：`limit` 在非 tight 时是 1 不是 9（这是二进制）；n = 0 要单独处理
  二进制串（`bin(0)` 是 `"0b0"`，用 `bin(n)[2:]` 得到 `"0"`）；返回值用整型即可。
- **相似题**：2801（模式三末题）把「相邻位关系」从「不能都填 1」推广到「数字必须
  相差 1」，状态从布尔量升级为「上一位数字」。

### 1215. 步进数（中等）

**题目**：步进数是相邻两位数字都正好相差 1 的数。给定 low、high，按升序返回
`[low, high]` 内所有步进数。

**思路**：

```python
from collections import deque


def stepping_numbers(low, high):
    result = []
    if low <= 0 <= high:
        result.append(0)
    q = deque(range(1, 10))
    while q:
        x = q.popleft()
        if x > high:
            continue
        if x >= low:
            result.append(x)
        last = x % 10
        if last > 0:
            q.append(x * 10 + last - 1)
        if last < 9:
            q.append(x * 10 + last + 1)
    result.sort()
    return result
```

```cpp
std::vector<int> steppingNumbers(int low, int high) {
    std::vector<int> result;
    if (low <= 0 && 0 <= high) {
        result.push_back(0);
    }
    std::queue<long long> q;
    for (int i = 1; i <= 9; ++i) {
        q.push(i);
    }
    while (!q.empty()) {
        long long x = q.front();
        q.pop();
        if (x > high) {
            continue;
        }
        if (x >= low) {
            result.push_back(static_cast<int>(x));
        }
        int last = static_cast<int>(x % 10);
        if (last > 0) {
            q.push(x * 10 + last - 1);
        }
        if (last < 9) {
            q.push(x * 10 + last + 1);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
```

**为什么用「生长」而不是数位 DP**：本题要的是**具体的数**（一个列表），不是个数。
与其逐位判断合法性，不如反过来主动生长：一个一位数（1..9）末尾接上「末位 ± 1」
就得到一个新的步进数。用 BFS 不断生长，落在 `[low, high]` 里的收集起来。

注意剪枝：`x > high` 时它的所有后代只会更大，可直接跳过；`last > 0` / `last < 9`
保证接出来的还是合法数字（0..9）。生长顺序不保证升序，最后统一排序。0 不在
「1..9 起点」的生成里，单独判断。

- **复杂度**：时间 O(答案个数)，空间 O(答案个数)。
- **易错点**：`x * 10 + ...` 用 `long long`，别在生成过程中溢出 int；`last == 0`
  时不能接 `-1`，`last == 9` 时不能接 `10`；0 要单独处理（它是步进数吗？本题
  按 LeetCode 语义，范围含 0 时 0 也算）。
- **相似题**：2801（下题）问的是同一个条件下「有多少个」，但上界大到只能数个数；
  600（上题）是相邻位约束的另一种形态。

### 2801. 统计范围内的步进数字数目（困难）

**题目**：步进数字指相邻两位数字都正好相差 1 的数字（如 10、12、321）。给定两个
字符串 low、high（长度可达 100），返回 `[low, high]` 内步进数字的个数，对 1e9+7
取模。

**思路**：

```python
from functools import lru_cache

MOD = 10 ** 9 + 7


def _decrement(s):
    arr = list(s)
    i = len(arr) - 1
    while arr[i] == "0":
        arr[i] = "9"
        i -= 1
    arr[i] = chr(ord(arr[i]) - 1)
    trimmed = "".join(arr).lstrip("0")
    return trimmed if trimmed else "0"


def _count_upto(s):
    length = len(s)

    @lru_cache(maxsize=None)
    def dfs(pos, prev, started, tight):
        if pos == length:
            return 1 if started else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            ntight = tight and d == limit
            if not started and d == 0:
                total += dfs(pos + 1, -1, False, ntight)
            elif started and abs(d - prev) != 1:
                continue
            else:
                total += dfs(pos + 1, d, True, ntight)
        return total % MOD

    return dfs(0, -1, False, True)


def count_stepping_numbers(low, high):
    return (_count_upto(high) - _count_upto(_decrement(low))) % MOD
```

```cpp
const long long MOD = 1000000007LL;
std::string s;
int length;
long long memo[105][11][2][2];

long long dfs(int pos, int prev, bool started, bool tight) {
    if (pos == length) {
        return started ? 1 : 0;
    }
    long long &res = memo[pos][prev + 1][started][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, -1, false, ntight);
        } else if (started && d - prev != 1 && prev - d != 1) {
            continue;
        } else {
            total += dfs(pos + 1, d, true, ntight);
        }
    }
    return res = total % MOD;
}

long long countUpto(const std::string &x) {
    s = x;
    length = static_cast<int>(x.size());
    std::memset(memo, -1, sizeof(memo));
    return dfs(0, -1, false, true);
}

std::string decrement(std::string a) {
    int i = static_cast<int>(a.size()) - 1;
    while (a[i] == '0') {
        a[i] = '9';
        --i;
    }
    a[i] = static_cast<char>(a[i] - 1);
    std::size_t start = a.find_first_not_of('0');
    if (start == std::string::npos) {
        return "0";
    }
    return a.substr(start);
}

int countSteppingNumbers(std::string low, std::string high) {
    long long r = (countUpto(high) - countUpto(decrement(low))) % MOD;
    if (r < 0) {
        r += MOD;
    }
    return static_cast<int>(r);
}
```

**和 1215 的关系**：条件完全相同，区别在上界——这里 high 长度可达 100，步进数的
个数多到数不清，不能生长枚举，只能「数个数」，于是回到数位 DP。

状态是「上一位数字 `prev`」：填下一位 `d` 时，若已开始则要求 `abs(d - prev) == 1`。
`prev = -1` 表示还没填（对应 `started` 为假）；`started` 为假且这一位填 0 时仍是
前导零，状态保持不动。走到串尾只有 `started` 为真才算一个步进数（排除全 0）。
区间答案仍是 `count_upto(high) - count_upto(low-1)`，因为 low 是大数，减一按字符串
处理：从末位借位，把末尾的 0 变成 9，再让第一个非零位减一，最后去掉前导零。

- **复杂度**：时间 O(位数 × 10 × 2 × 2)，空间同阶。
- **易错点**：取模后相减可能为负，要 `+MOD`；`decrement` 全为 0 的边界不适用
  （low ≥ 1），但仍要能返回 `"0"` 兜底；C++ 比较「相差 1」时不要用 `std::abs`
  多引头文件，写成 `d - prev != 1 && prev - d != 1` 即可；`memo` 的下标 `prev + 1`
  把 -1 映射到 0。
- **相似题**：1215（上题）是它的枚举版；600（上上题）同样是相邻位约束。

---

## 模式四：固定数字集与数位和

**适用信号**：每一位能填的数字被限制在一个给定集合里（902），或者条件是关于
「所有位数字之和」的区间（2719）。

**核心动作**：前者在枚举每一位时只遍历给定集合；后者在状态里多带一个「已累计的
数位和」，并用上界 `max_sum` 剪枝。

### 902. 最大为 N 的数字组合（困难）

**题目**：给定一个升序、不含重复元素且只含 '1'..'9' 的数字字符数组 digits，以及
整数 n，返回用 digits 里的数字（每个可重复使用）能拼出的、数值 <= n 的正整数
个数。

**思路**：

```python
from functools import lru_cache


def at_most_n_given_digit_set(digits, n):
    nums = [int(c) for c in digits]
    s = str(n)
    length = len(s)

    total = 0
    for size in range(1, length):
        total += len(nums) ** size

    @lru_cache(maxsize=None)
    def dfs(pos, tight):
        if pos == length:
            return 1
        limit = int(s[pos]) if tight else 9
        count = 0
        for d in nums:
            if d > limit:
                break
            count += dfs(pos + 1, tight and d == limit)
        return count

    return total + dfs(0, True)
```

```cpp
std::string s;
int length;
std::vector<int> nums;
long long memo[12][2];

long long dfs(int pos, bool tight) {
    if (pos == length) {
        return 1;
    }
    long long &res = memo[pos][tight];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long count = 0;
    for (int d : nums) {
        if (d > limit) {
            break;
        }
        count += dfs(pos + 1, tight && d == limit);
    }
    return res = count;
}

long long atMostNGivenDigitSet(std::vector<std::string> &digits, int n) {
    nums.clear();
    for (const std::string &c : digits) {
        nums.push_back(c[0] - '0');
    }
    s = std::to_string(n);
    length = static_cast<int>(s.size());
    long long total = 0;
    for (int size = 1; size < length; ++size) {
        long long power = 1;
        for (int i = 0; i < size; ++i) {
            power *= static_cast<long long>(nums.size());
        }
        total += power;
    }
    std::memset(memo, -1, sizeof(memo));
    return total + dfs(0, true);
}
```

**为什么按位数分成两段**：因为数字集里没有 0，用它们拼出的数**天然没有前导零、
也不含 0**，于是「前导零」这个麻烦直接消失了。

设 n 有 L 位：

1. 位数比 L 少的数一定小于 n，长度为 `size` 时有 `len(nums)^size` 种，长度
   1..L-1 全部加进来；
2. 位数恰好为 L 的数再逐位和 n 比较：`tight` 时这一位的上限是 `s[pos]`，否则是 9；
   从有序的 `nums` 里依次取不超过上限的数字，一旦超上限，后面更大，直接 `break`。

两段相加即可。这里不需要 `started`，正是「数字集排除 0」带来的便利。

- **复杂度**：时间 O(位数 × |digits|)，空间 O(位数 × |digits|)。
- **易错点**：长度为 `size` 的个数是 `len(nums) ** size`，注意别把位数算错；用
  `long long` 存累加；C++ 里传入的是 `vector<string>`，取 `c[0] - '0'`。
- **相似题**：2719（下题）的「每位只能从某集合取值」感觉相近，但它的约束在
  「数位和」上；357 的闭式里也用了「每选一位少一个可用数字」的乘法原理。

### 2719. 统计整数数目（困难）

**题目**：给定两个数字字符串 num1、num2 和两个整数 min_sum、max_sum，统计
`[num1, num2]` 内「各位数字之和落在 `[min_sum, max_sum]`」的整数个数，对 1e9+7
取模。

**思路**：

```python
from functools import lru_cache

MOD = 10 ** 9 + 7


def _decrement(s):
    arr = list(s)
    i = len(arr) - 1
    while arr[i] == "0":
        arr[i] = "9"
        i -= 1
    arr[i] = chr(ord(arr[i]) - 1)
    trimmed = "".join(arr).lstrip("0")
    return trimmed if trimmed else "0"


def count_of_integers(num1, num2, min_sum, max_sum):
    def count_upto(s):
        length = len(s)

        @lru_cache(maxsize=None)
        def dfs(pos, sum_so_far, tight, started):
            if pos == length:
                return 1 if started and min_sum <= sum_so_far <= max_sum else 0
            limit = int(s[pos]) if tight else 9
            total = 0
            for d in range(limit + 1):
                ntight = tight and d == limit
                if not started and d == 0:
                    total += dfs(pos + 1, sum_so_far, ntight, False)
                else:
                    if sum_so_far + d > max_sum:
                        continue
                    total += dfs(pos + 1, sum_so_far + d, ntight, True)
            return total % MOD

        return dfs(0, 0, True, False)

    return (count_upto(num2) - count_upto(_decrement(num1))) % MOD
```

```cpp
const long long MOD = 1000000007LL;
std::string s;
int length;
int minSum;
int maxSum;
long long memo[25][405][2][2];

long long dfs(int pos, int sum, bool tight, bool started) {
    if (pos == length) {
        return (started && sum >= minSum && sum <= maxSum) ? 1 : 0;
    }
    long long &res = memo[pos][sum][tight][started];
    if (res != -1) {
        return res;
    }
    int limit = tight ? s[pos] - '0' : 9;
    long long total = 0;
    for (int d = 0; d <= limit; ++d) {
        bool ntight = tight && (d == limit);
        if (!started && d == 0) {
            total += dfs(pos + 1, sum, ntight, false);
        } else {
            if (sum + d > maxSum) {
                continue;
            }
            total += dfs(pos + 1, sum + d, ntight, true);
        }
    }
    return res = total % MOD;
}

long long countUpto(const std::string &x) {
    s = x;
    length = static_cast<int>(x.size());
    std::memset(memo, -1, sizeof(memo));
    return dfs(0, 0, true, false);
}

std::string decrement(std::string a) {
    int i = static_cast<int>(a.size()) - 1;
    while (a[i] == '0') {
        a[i] = '9';
        --i;
    }
    a[i] = static_cast<char>(a[i] - 1);
    std::size_t start = a.find_first_not_of('0');
    if (start == std::string::npos) {
        return "0";
    }
    return a.substr(start);
}

int countOfIntegers(std::string num1, std::string num2, int _minSum, int _maxSum) {
    minSum = _minSum;
    maxSum = _maxSum;
    long long r = (countUpto(num2) - countUpto(decrement(num1))) % MOD;
    if (r < 0) {
        r += MOD;
    }
    return static_cast<int>(r);
}
```

**状态为什么多一维**：条件「数位和落在区间」在填完之前无法判断，所以状态里必须
带着「已累计的数位和 `sum_so_far`」。填新数字 `d` 时先看 `sum_so_far + d` 有没有
超过 `max_sum`，超了这条分支直接砍掉（剪枝，也让状态空间更小）；低于 `min_sum`
的情况留到最后统一判断——因为后面还可能加上来。

`started` 依然用来排除前导零：本题 min_sum ≥ 1，0 的数位和是 0，本来就不合格，
但为了逻辑干净还是加上 `started`，走到串尾要求「已开始且和落在区间内」。区间答案
同样是 `count_upto(num2) - count_upto(num1-1)`，减一按字符串借位。

- **复杂度**：时间 O(位数 × max_sum × 2 × 2)，空间同阶。
- **易错点**：`sum + d > max_sum` 的剪枝针对的是上界，别错误地对下界提前剪枝；
  取模相减要为负补 `MOD`；`memo` 的 sum 维开到 `max_sum + 1` 以上（本题 400 上限
  取 405 足够）；递归深度可达 num2 长度（≤ 22 或更长），Python 默认递归上限够用。
- **相似题**：902（上题）限制的是「每位能取哪些数字」，本题限制的是「和的范围」；
  233 / 1067 的「按位统计」与「累计量放进状态」是同一个思想的两种走向。

---

## 规律总结

1. **数位 DP 的三件套：前缀相减、`tight`、`started`**。区间问题先用
   `f(R) - f(L-1)` 压成上界问题；`tight` 控制当前位是否被上界约束；`started`
   处理前导零（也让「正数」与「包含 0」按题意取舍）。

2. **状态怎么定，看条件依赖什么**。只依赖「这一位是否用过」→ 用掩码（1012 /
   2376）；只依赖「上一位」→ 记上一位（600 / 2801）；依赖累计信息 → 记数位和
   （2719）；依赖「能取哪些数字」→ 直接改枚举集合（902）。状态定义是数位 DP
   唯一的难点，其余都是模板。

3. **`tight` 的转移永远是 `tight && (d == limit)`**。贴着上界的分支只有一条，
   所以它对复杂度几乎没有贡献，把它一起丢进记忆化也不会爆状态。

4. **能闭式就别记忆化**。当上界是 10 的幂、条件又只跟「用过几个不同数字」有关时
   （357），掩码会被压缩成一个「还剩几种可选」的计数，直接乘法原理 O(位数) 解决。

5. **相邻位约束状态最简单**。600 只记一个布尔（上一位是不是 1），2801 只记一个
   数字（上一位是几）。这类题的合法性在「填下一位的那一刻」就能判定，剪枝也顺手
   做了。

6. **大数字符串上的「减一」要自己做**。从末位借位：末尾的 0 变 9，第一个非零位
   减一，最后去掉前导零。low 是大数时（2801 / 2719）必须先减一再做前缀相减。

7. **取模题注意负差**。`(f(high) - f(low-1)) % MOD` 在语言里可能得到负数，
   统一 `if (r < 0) r += MOD`。

8. **前导零要用 `started` 单独隔离**。`not started and d == 0` 时状态必须原样
   传递，既不占用数字掩码，也不参与相邻位比较，更不算「已开始」。这一条写错，
   答案往往只差一点点，特别难查。

9. **该剪枝就剪枝**。数位和超过上界（2719）直接 `continue`；数字集有序时超过
   当前上限直接 `break`（902）；BFS 生成时超过 high 不再生长（1215）。剪枝
   既省时间，也让状态空间可预估。

10. **与其它篇的联系**：区间用前缀相减，是第 04 篇「前缀和」的思想；
    掩码记录「用过哪些数字」是第 16 篇「位运算」的典型用法；「逐位填、把重复状态
    记下来」本质上就是第 13 篇「动态规划」的记忆化搜索，只是状态长在十进制串上。
    数位 DP 可以说是「带前导零与上界处理的记忆化搜索」，把这三篇的招式缝在了一起。
