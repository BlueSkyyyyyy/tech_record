# 排序：从「会用排序」到「定制排序规则」

很多题看起来和排序无关，但最优解的骨架就是「先排个序，剩下的问题就变简单了」。排序在这里
不是目的，而是一件工具：**把无序变有序之后，原本要两两比较的信息，往往只需看相邻几项**。
更进阶一点，有些题要排的既不是数字大小、也不是字典序，而是**题目自定义的一套顺序**（比如
「按出现频率排」「哪两个数拼起来更大」），这就需要我们学会给排序**定制比较规则**。

本篇用十道题覆盖排序类题目的几条主线：直接利用已有的有序性（977），排完后回填名次（506），
按自定义顺序排（1122），按频率排（1636、451），用自定义比较器排（179），把排序当「统计上界」
的工具（274），只分三类的原地排序（75），链表上的排序（147），以及不走比较排序、用桶和鸽巢
原理做到线性的排序（164）。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：利用已有的有序性 | 977. 有序数组的平方 | 简单 |
| 模式二：排序后回填「名次/位置」 | 506. 相对名次 | 简单 |
| 模式三：按「自定义顺序」排序 | 1122. 数组的相对排序 | 简单 |
| 模式四：按「出现频率」排序 | 1636. 按照频率将数组升序排序 / 451. 根据字符出现频率排序 | 简单/中等 |
| 模式五：自定义比较器 | 179. 最大数 | 中等 |
| 模式六：拿排序当「上界」工具 | 274. H 指数 | 中等 |
| 模式七：原地三路划分（荷兰国旗） | 75. 颜色分类 | 中等 |
| 模式八：链表上的插入排序 | 147. 对链表进行插入排序 | 中等 |
| 模式九：桶排序与鸽巢原理 | 164. 最大间距 | 困难 |

读这一篇时，重点不是背某个排序算法，而是建立两种判断：**这道题该不该排序、按什么键排序**。
只要想清楚「比较规则」，代码往往就是一句语言自带的排序调用；而当题目要求线性时间时，再
想想「数值范围」或「频率范围」是否有限——有限就可以用计数/桶排序替代比较排序。

---

## 模式一：利用已有的有序性

**适用信号**：数组本身已经有序，但某个操作（如平方、取绝对值）会破坏有序性，需要重新组织
结果。

**核心动作**：不要无视「已有序」这条现成信息去重新排序，而是找出「最大值一定在两端」这类
性质，用双指针从两端向中间收。

### 977. 有序数组的平方（简单）

**题目**：给你一个按非递减顺序排序的整数数组 `nums`（可以含负数），返回每个数字的平方
组成的新数组，要求同样按非递减顺序排列。

**思路**：

平方会抹掉符号：越靠近两端的数，绝对值越大，平方也越大。数组已经有序，所以「绝对值最大」
的元素一定在两端之一，不可能藏在中间。

于是用左右两个指针 `l`、`r` 指向当前未处理区间的两端，每轮比较 `nums[l]`、`nums[r]` 的
绝对值，把较大者平方后从结果数组的**末尾**往前放，再让对应指针向内收一格。

**为什么从末尾往前填**：两端拿到的都是「当前最大」的平方，最大的数应该排在结果最后，所以
从后往前写正合适。**为什么不能直接平方再排序**：那样是 O(n log n)，而双指针利用了原数组
有序这条信息，只要 O(n)。

**代码**（完整可运行版见 `src/sort/sorted_squares.py` / `.cpp`）：

```python
def sorted_squares(nums):
    n = len(nums)
    res = [0] * n
    l, r = 0, n - 1
    k = n - 1
    while l <= r:
        if abs(nums[l]) > abs(nums[r]):
            res[k] = nums[l] * nums[l]
            l += 1
        else:
            res[k] = nums[r] * nums[r]
            r -= 1
        k -= 1
    return res
```

```cpp
std::vector<int> sortedSquares(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<int> res(n, 0);
    int l = 0, r = n - 1, k = n - 1;
    while (l <= r) {
        if (std::abs(nums[l]) > std::abs(nums[r])) {
            res[k] = nums[l] * nums[l];
            ++l;
        } else {
            res[k] = nums[r] * nums[r];
            --r;
        }
        --k;
    }
    return res;
}
```

- **复杂度**：时间 O(n)，空间 O(n)（结果数组）。比较时用绝对值，避免手写正负号判断。
- **易错点**：左闭右闭区间，循环条件是 `l <= r`，别写成 `<`，否则中间那个元素会漏掉；
  相等时两个指针随便收哪个都行，但要保证每轮都推进，否则死循环；`k` 从 `n-1` 递减。
- **相似题**：88. 合并两个有序数组（`docs/01-array-two-pointers.md`，从后往前填，与本题
  同一套「从大端开始放」的写法）；360 有序转化数组（同族的正负平方变形）。

---

## 模式二：排序后回填「名次/位置」

**适用信号**：答案与「排好序之后的位置」有关，但输出必须按原始下标排列。

**核心动作**：把**下标**按对应值排序，得到「第几名是谁」的名单，再按下标把结果写回。

### 506. 相对名次（简单）

**题目**：给你长度为 n 的数组 `score`，`score[i]` 是第 i 位运动员的成绩，所有成绩互不
相同。返回答案数组，其中第 i 位是：前三名依次为 `"Gold Medal"`、`"Silver Medal"`、
`"Bronze Medal"`，其余为名次数字的字符串（成绩最高者名次为 1）。

**思路**：

名次就是「按成绩从大到小排完后的位置」，但答案要按运动员原本的下标排列。做法是：先把
**下标数组**按成绩降序排序，得到 `order`（`order[0]` 是成绩最高者的下标）；再遍历
`order`，第 `k` 个位置（`k` 从 0 数起）对应的名次是 `k+1`，把名次写回 `res[order[k]]`。

**为什么排下标而不是排名次对象**：排序键是成绩，但最终要的是原下标处的答案，让下标跟着
成绩一起排，排完仍能通过 `order` 找回原位置。

**代码**（`src/sort/relative_ranks.py` / `.cpp`）：

```python
def find_relative_ranks(score):
    order = sorted(range(len(score)), key=lambda i: -score[i])
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    res = [""] * len(score)
    for rank, i in enumerate(order):
        res[i] = medals[rank] if rank < 3 else str(rank + 1)
    return res
```

```cpp
std::vector<std::string> findRelativeRanks(const std::vector<int> &score) {
    int n = static_cast<int>(score.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&score](int a, int b) { return score[a] > score[b]; });
    std::vector<std::string> medals = {"Gold Medal", "Silver Medal", "Bronze Medal"};
    std::vector<std::string> res(n);
    for (int rank = 0; rank < n; ++rank) {
        int i = order[rank];
        res[i] = rank < 3 ? medals[rank] : std::to_string(rank + 1);
    }
    return res;
}
```

- **复杂度**：时间 O(n log n)（排序），空间 O(n)（下标数组与答案）。
- **易错点**：`res` 的下标是「运动员编号」，不要错写成 `order` 的下标；名次从 1 开始，
  前三名的判断用 `rank < 3`；`-score[i]` 是最简便的「降序」键。
- **相似题**：1122（下面，按另一套顺序排）；1636（按频率排后再回填）。

---

## 模式三：按「自定义顺序」排序

**适用信号**：题目先给一个「参考顺序」数组，要求按它来排另一个数组。

**核心动作**：把「自定义顺序」翻译成一个可比较的**排序键**——参考数组里的下标就是优先级，
不在参考里的给一个统一的末尾优先级，再配合数值本身升序。

### 1122. 数组的相对排序（简单）

**题目**：给你数组 `arr1` 和 `arr2`。`arr2` 中元素互不相同，且都出现在 `arr1` 中。请把
`arr1` 排序，使其中元素的相对顺序与 `arr2` 一致；未在 `arr2` 中出现的元素，按升序排在末尾。

**思路**：

这类「自定义顺序」的通用技巧，是给每个值造一个排序键：某个值在 `arr2` 里的**下标**就是
它的优先级，越小越靠前；不在 `arr2` 里的值统一给一个大优先级（比如 `len(arr2)`），它们
之间再用数值本身升序打破平局。键写成二元组 `(优先级, 数值)`，直接交给语言自带排序。

**为什么一行就够**：元组比较天然按字典序逐级进行：先比优先级，优先级相同再比数值，恰好
把「先按 arr2、再按升序」两条规则合并了。

另一种更快的写法是**计数排序**：先用桶统计 `arr1` 各值出现次数，按 `arr2` 顺序输出对应
次数，再把剩下的非零值按升序输出，时间 O(n + m + 数值范围)。数值范围大时比较排序更通用，
范围小则计数更快。

**代码**（`src/sort/relative_sort_array.py` / `.cpp`）：

```python
def relative_sort_array(arr1, arr2):
    rank = {v: i for i, v in enumerate(arr2)}
    return sorted(arr1, key=lambda x: (rank.get(x, len(arr2)), x))
```

```cpp
std::vector<int> relativeSortArray(const std::vector<int> &arr1,
                                   const std::vector<int> &arr2) {
    std::unordered_map<int, int> rank;
    for (int i = 0; i < static_cast<int>(arr2.size()); ++i) {
        rank[arr2[i]] = i;
    }
    int fallback = static_cast<int>(arr2.size());
    std::vector<int> res = arr1;
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        int ra = rank.count(a) ? rank[a] : fallback;
        int rb = rank.count(b) ? rank[b] : fallback;
        if (ra != rb) {
            return ra < rb;
        }
        return a < b;
    });
    return res;
}
```

- **复杂度**：比较排序时间 O(n log n)、空间 O(m)。计数排序可降到 O(n + m + range)。
- **易错点**：不在 `arr2` 里的值一定要给**统一且最大**的优先级，且彼此之间还要按数值升序，
  否则末尾那段的顺序会是乱的；`rank.get(x, len(arr2))` 的默认值别漏。
- **相似题**：1636（按频率这一「隐式顺序」排）；179（比较规则更怪的自定义排序）。

---

## 模式四：按「出现频率」排序

**适用信号**：排序依据不是元素大小，而是「这个元素出现了几次」。

**核心动作**：先统计频率，再用「频率、元素」组成排序键；或者用「频率当桶下标」做桶排序。

### 1636. 按照频率将数组升序排序（简单）

**题目**：给你整数数组 `nums`，按每个值出现的频率升序排序；频率相同时按数值降序排列。

**思路**：

先把频率数出来（用计数器），然后对整个数组排序，键写成 `(cnt[x], -x)`：先比频率升序；
频率相同比 `-x`，即数值大的在前。**为什么直接排原数组**：同一个值要重复它应有的次数，
原数组恰好是「每个值展开了若干次」的现成载体，按上述键排序即可。

**代码**（`src/sort/frequency_sort.py` / `.cpp`）：

```python
def frequency_sort(nums):
    from collections import Counter

    cnt = Counter(nums)
    return sorted(nums, key=lambda x: (cnt[x], -x))
```

```cpp
std::vector<int> frequencySort(const std::vector<int> &nums) {
    std::unordered_map<int, int> cnt;
    for (int x : nums) {
        ++cnt[x];
    }
    std::vector<int> res = nums;
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        if (cnt[a] != cnt[b]) {
            return cnt[a] < cnt[b];
        }
        return a > b;
    });
    return res;
}
```

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：第二关键字是「数值降序」，别顺手写成升序；负数用 `-x` 依然正确（`-x` 小
  代表 `x` 大）。
- **相似题**：451（同一思路，排的是字符）。

### 451. 根据字符出现频率排序（中等）

**题目**：给你字符串 `s`，按字符出现频率降序重新排列，返回排序后的字符串；同频字符顺序不限。

**思路**：

目标是「频率高的字符排前面」，且字符要重复它出现的次数。先用计数器统计每个字符出现几次；
把「字符 → 次数」这些项按次数降序排；再把每项展开成「该字符重复次数遍」拼起来。

**为什么不用对原字符串直接排序**：原字符串里字符已经带着应有的重复次数，但同频字符的相对
次序会依赖稳定排序的输入顺序，不如直接按 `(字符 → 次数)` 构造清晰可控。**最快的写法**是
桶排序：频率最大不超过 n，开 n+1 个桶把字符按频率放进去，再从高频率往低频率输出，时间
O(n)。

**代码**（`src/sort/sort_characters_by_frequency.py` / `.cpp`，为结果确定，同频按字符升序）：

```python
def frequency_sort_string(s):
    from collections import Counter

    cnt = Counter(s)
    parts = [ch * c for ch, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    return "".join(parts)
```

```cpp
std::string frequencySort(std::string s) {
    std::unordered_map<char, int> cnt;
    for (char ch : s) {
        ++cnt[ch];
    }
    std::vector<std::pair<char, int>> items(cnt.begin(), cnt.end());
    std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });
    std::string res;
    for (const auto &p : items) {
        res.append(static_cast<size_t>(p.second), p.first);
    }
    return res;
}
```

- **复杂度**：比较排序时间 O(n log k)（k 为不同字符数，k ≤ n），空间 O(k)；桶排序可
  做到 O(n)。
- **易错点**：`ch * c` 里 `c` 是次数，别把次数当成字符；空串要能返回空串（`parts` 为空，
  拼接结果自然为空）。
- **相似题**：1636（同思路的整数版）；347. 前 K 个高频元素（`docs/08-heap.md`，不过只取
  前 K 个时用堆更省）。

---

## 模式五：自定义比较器

**适用信号**：两个元素谁排前面，不能用大小或字典序直接判断，而要「试着拼一下看谁更优」。

**核心动作**：定义一个比较函数（或比较键），交给自己实现不了的场景，让语言按你的规则排序。

### 179. 最大数（中等）

**题目**：给你非负整数数组 `nums`，重新排列后拼成一个数，要求拼出的数最大，以字符串返回。

**思路**：

把大的数字放前面并不对：比如 3 和 30，数字上 3 小，但拼成 `"330"` 比 `"303"` 大，所以 3
应排在 30 前面。可见排序依据是「两个数谁放前面能拼出更大的结果」。

对任意两个数 `a`、`b`，比较拼法 `a+b` 与 `b+a`：若 `a+b > b+a`，就规定 a 排在 b 前面。
这个关系满足全序，可以用它当比较器对整个数字串排序，再依次拼接。

**为什么这个比较器对整体最优有效（交换论证）**：拼接结果可看成所有数字串按某个顺序首尾相接。
若存在相邻两项 `a`、`b` 使 `a+b < b+a`，交换它们会让整体结果变大（其余部分不变），所以
最优排列里不存在这样的逆序对。于是「任意相邻都满足 `a+b >= b+a`」的排列就是最优，而这
正是按上述比较器排序得到的结果。

**代码**（`src/sort/largest_number.py` / `.cpp`）：

```python
def largest_number(nums):
    from functools import cmp_to_key

    def cmp(a, b):
        if a + b > b + a:
            return -1
        if a + b < b + a:
            return 1
        return 0

    strs = [str(x) for x in nums]
    strs.sort(key=cmp_to_key(cmp))
    res = "".join(strs)
    return "0" if res[0] == "0" else res
```

```cpp
std::string largestNumber(const std::vector<int> &nums) {
    std::vector<std::string> strs;
    strs.reserve(nums.size());
    for (int x : nums) {
        strs.push_back(std::to_string(x));
    }
    std::sort(strs.begin(), strs.end(),
              [](const std::string &a, const std::string &b) {
                  return a + b > b + a;
              });
    std::string res;
    for (const auto &s : strs) {
        res += s;
    }
    if (!res.empty() && res[0] == '0') {
        return "0";
    }
    return res;
}
```

- **复杂度**：时间 O(n log n * L)，其中 L 是数字串平均长度（每次比较要做一次字符串拼接）；
  空间 O(n)。
- **易错点**：比较器必须满足严格弱序（`a+b > b+a` 就是，不要写成 `>=`，否则排序行为未
  定义）；**前导零**：全是 0 时结果应返回 `"0"` 而不是 `"000"`；`a+b` 与 `b+a` 一定等长，
  所以字典序比较和数值比较一致。
- **相似题**：1122（把顺序规则简化为优先级键）；973. 最接近原点的 K 个点（`docs/08-heap.md`，
  自定义比较键的另一例）。

---

## 模式六：拿排序当「上界」工具

**适用信号**：要判断「至少有多少个元素满足某个阈值条件」，且最优解对应排序后的某个前缀。

**核心动作**：降序排序，让「前 i 个都至少是多少」变得显然，再线性扫描找到临界点。

### 274. H 指数（中等）

**题目**：给你数组 `citations`，`citations[i]` 是第 i 篇论文被引用次数。H 指数定义为
满足「至少有 h 篇论文每篇被引用至少 h 次」的最大整数 h。

**思路**：

把引用次数从大到小排序。排完后第 i 个位置（从 1 数起）表示「前 i 篇里引用最少的那篇」。
若 `citations[i-1] >= i`，说明前 i 篇每篇至少被引 i 次，h 至少能取到 i；继续往后直到条件
第一次不成立就停。

**为什么排序后只扫一遍就够**：排序把「任意 h 篇」变成了「引用最多的前 h 篇」。要凑出 h 篇
都至少 h 次，显然应挑引用最多的前 h 篇；连它们都不够，别的组合更不可能。所以最优 h 一定
对应排序后的某个前缀。

也可以不排序：用计数数组统计「引用次数为 k 的论文有几篇」（k 超过 n 的按 n 记），从高到低
累加篇数，累计篇数首次达到当前引用次数时即得答案，时间 O(n)。

**代码**（`src/sort/h_index.py` / `.cpp`）：

```python
def h_index(citations):
    citations.sort(reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h
```

```cpp
int hIndex(std::vector<int> citations) {
    std::sort(citations.begin(), citations.end(), std::greater<int>());
    int h = 0;
    for (int i = 0; i < static_cast<int>(citations.size()); ++i) {
        if (citations[i] >= i + 1) {
            h = i + 1;
        } else {
            break;
        }
    }
    return h;
}
```

- **复杂度**：排序解时间 O(n log n)、空间 O(1)（原地排序）；计数解时间 O(n)、空间 O(n)。
- **易错点**：下标从 1 数起，判断是 `citations[i] >= i + 1`；命中条件后要更新 `h`，不能
  一遇到不满足就直接返回上一轮以外的值；空数组返回 0。
- **相似题**：164（下面，也是「排序后看相邻/位置」）；506（排序后按位置定名次）。

---

## 模式七：原地三路划分（荷兰国旗）

**适用信号**：元素只有有限的几种取值，要求原地、一趟、常数空间排好序。

**核心动作**：不用通用排序，开三个指针把数组分成「小值区 / 中值区 / 大值区」，一次扫描
完成归位。

### 75. 颜色分类（中等）

**题目**：给定含 0、1、2 的数组 `nums`，原地排序使相同颜色相邻、按 0、1、2 顺序排列。
不用库排序，只允许常数级额外空间。

**思路**：

只有三种取值，可以一次扫描把它们分到三块。用三个指针：`p0` 是 0 区右边界之后的位置
（下一个 0 该放的地方），`p2` 是 2 区左边界之前的位置，`cur` 是当前考察位置。从头扫描：

- 遇到 0：和 `p0` 交换，`p0`、`cur` 都右移；
- 遇到 2：和 `p2` 交换，`p2` 左移，但 `cur` **不动**；
- 遇到 1：`cur` 右移。

**为什么遇到 0 后 cur 能前进、遇到 2 后不能**：遇到 0 时换到 `cur` 位置的是 `p0` 处的
元素，而 `p0 <= cur`，`p0` 走过的地方只可能是已归位的 0 或就是自己，换来的元素是安全的，
所以 `cur` 可以前进。遇到 2 时，从 `p2` 换过来的是**还没看过**的元素，它可能是 0、1、2，
必须留在原地下一轮再判，所以 `cur` 不动。

**为什么一趟就够**：`cur` 只在处理 0/1 时前进，处理 2 时靠 `p2` 收缩，`cur > p2` 时中间
全部归位，无需第二趟。

**代码**（`src/sort/sort_colors.py` / `.cpp`）：

```python
def sort_colors(nums):
    p0, cur, p2 = 0, 0, len(nums) - 1
    while cur <= p2:
        if nums[cur] == 0:
            nums[p0], nums[cur] = nums[cur], nums[p0]
            p0 += 1
            cur += 1
        elif nums[cur] == 2:
            nums[cur], nums[p2] = nums[p2], nums[cur]
            p2 -= 1
        else:
            cur += 1
```

```cpp
void sortColors(std::vector<int> &nums) {
    int p0 = 0, cur = 0, p2 = static_cast<int>(nums.size()) - 1;
    while (cur <= p2) {
        if (nums[cur] == 0) {
            std::swap(nums[p0], nums[cur]);
            ++p0;
            ++cur;
        } else if (nums[cur] == 2) {
            std::swap(nums[cur], nums[p2]);
            --p2;
        } else {
            ++cur;
        }
    }
}
```

- **复杂度**：时间 O(n)（每个元素最多被处理一次），空间 O(1)。
- **易错点**：交换 2 之后 `cur` **不能自增**，否则会把换过来的未知元素跳过去；`p2` 左移
  后可能小于 `cur`，循环条件要写 `cur <= p2`。
- **相似题**：215. 数组中的第 K 个最大元素（`docs/08-heap.md`，快速选择里的分区同源）；
  26/27/80（`docs/01-array-two-pointers.md`，都是「一趟原地划分」的思想）。

---

## 模式八：链表上的插入排序

**适用信号**：待排序的数据是链表，只能靠改指针，不能随机访问。

**核心动作**：维护一条「已排好序」的链，用虚拟头结点当哨兵，逐个把新节点插到正确位置。

### 147. 对链表进行插入排序（中等）

**题目**：给你单链表头节点 `head`，用**插入排序**将它升序排序，返回排序后的头节点。

**思路**：

用虚拟头结点 `dummy` 作为有序链的哨兵，这样「插到最前面」和「插到中间」可以用同一段代码
处理，不用为头结点写特例。每轮：

1. 用 `nxt` 记住当前节点的下一个（插入会改指针，先存后改，与反转链表同理）；
2. 从 `dummy` 出发，找第一个「值大于当前值」的节点，插在它前面；
3. 接好指针，游标移到 `nxt` 继续。

**为什么找的是「第一个更大」**：这样相等元素会排在已有相等元素之后，保持稳定，而链表插入
排序本身是稳定的。

**代码**（`src/sort/insertion_sort_list.py` / `.cpp`）：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def insertion_sort_list(head):
    dummy = ListNode(0)
    cur = head
    while cur:
        nxt = cur.next
        prev = dummy
        while prev.next and prev.next.val <= cur.val:
            prev = prev.next
        cur.next = prev.next
        prev.next = cur
        cur = nxt
    return dummy.next
```

```cpp
struct ListNode {
    int val;
    ListNode *next;
    explicit ListNode(int v = 0, ListNode *n = nullptr) : val(v), next(n) {}
};

ListNode *insertionSortList(ListNode *head) {
    ListNode dummy(0);
    ListNode *cur = head;
    while (cur) {
        ListNode *nxt = cur->next;
        ListNode *prev = &dummy;
        while (prev->next && prev->next->val <= cur->val) {
            prev = prev->next;
        }
        cur->next = prev->next;
        prev->next = cur;
        cur = nxt;
    }
    return dummy.next;
}
```

- **复杂度**：时间 O(n^2)（最坏每个节点都从头找插入点），空间 O(1)。
- **易错点**：必须先 `nxt = cur.next` 再改指针，否则后半段丢失；内层循环用 `<=` 保持稳定，
  且要先判 `prev->next` 非空；返回 `dummy.next` 而不是 `head`（头结点可能已经换人）。
- **相似题**：148. 排序链表（`docs/12-divide-conquer.md`，用归并做到 O(n log n)）；
  206. 反转链表（`docs/06-linked-list.md`，同样「先存后改」）。若追求更快，应改用归并。

---

## 模式九：桶排序与鸽巢原理

**适用信号**：要求**线性时间**，而直接比较排序是 O(n log n)；同时数值范围已知。

**核心动作**：把值域等分成若干桶，桶内只记最小/最大值；靠鸽巢原理证明「答案只可能跨桶」，
从而只扫桶、不排桶内元素。

### 164. 最大间距（困难）

**题目**：给定无序整数数组 `nums`，返回排序后相邻元素差值（间距）的最大值；元素少于 2 个
返回 0。要求线性时间。

**思路**：

设最小值 `lo`、最大值 `hi`、元素个数 n。把区间 `[lo, hi]` 等分成若干个桶，桶内只记录
出现过的**最小值和最大值**，桶宽取 `size = max(1, (hi - lo) // (n - 1))`，桶数为
`(hi - lo) // size + 1`。

**为什么最大间距一定跨桶（鸽巢原理）**：n 个数放进若干桶，若最大间距发生在同一桶内，则该
桶内两元素之差小于桶宽 `size`；但 `size` 不超过 n-1 个间距的平均值 `(hi-lo)/(n-1)`，于是
至少有一个间距不小于桶宽——矛盾。所以真正的最大间距必然出现在相邻两个非空桶的「后一桶
最小值 − 前一桶最大值」之间。

于是只扫一遍桶，维护「上一个非空桶的最大值 `prev`」（初始为 `lo`），用当前桶最小值减 `prev`
更新答案，再让 `prev` 变为当前桶最大值。

**代码**（`src/sort/maximum_gap.py` / `.cpp`）：

```python
def maximum_gap(nums):
    if len(nums) < 2:
        return 0
    n = len(nums)
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return 0
    size = max(1, (hi - lo) // (n - 1))
    count = (hi - lo) // size + 1
    buckets = [[None, None] for _ in range(count)]
    for x in nums:
        idx = (x - lo) // size
        b = buckets[idx]
        b[0] = x if b[0] is None else min(b[0], x)
        b[1] = x if b[1] is None else max(b[1], x)
    best = 0
    prev = lo
    for bmin, bmax in buckets:
        if bmin is None:
            continue
        best = max(best, bmin - prev)
        prev = bmax
    return best
```

```cpp
int maximumGap(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    if (n < 2) {
        return 0;
    }
    long long lo = *std::min_element(nums.begin(), nums.end());
    long long hi = *std::max_element(nums.begin(), nums.end());
    if (lo == hi) {
        return 0;
    }
    long long size = std::max(1LL, (hi - lo) / (n - 1));
    long long count = (hi - lo) / size + 1;
    const long long NONE = -1;
    std::vector<long long> bmin(count, NONE);
    std::vector<long long> bmax(count, NONE);
    for (int x : nums) {
        long long idx = (x - lo) / size;
        if (bmin[idx] == NONE || x < bmin[idx]) {
            bmin[idx] = x;
        }
        if (bmax[idx] == NONE || x > bmax[idx]) {
            bmax[idx] = x;
        }
    }
    int best = 0;
    long long prev = lo;
    for (long long i = 0; i < count; ++i) {
        if (bmin[i] == NONE) {
            continue;
        }
        best = std::max(best, static_cast<int>(bmin[i] - prev));
        prev = bmax[i];
    }
    return best;
}
```

- **复杂度**：时间 O(n)（建桶 + 扫桶各一趟），空间 O(n)（桶数组）。
- **易错点**：桶宽用整除可能得 0，必须 `max(1, ...)`；`lo == hi`（全相等）要提前返回 0，
  否则除零；`prev` 初始为 `lo`，保证第一个非空桶也能正确比较；相邻非空桶之间要跳过空桶。
- **相似题**：274（同上，排序当工具）；451（频率当桶下标，也是「范围有限就用桶」）；
  桶排序思想也见于 347. 前 K 个高频元素（`docs/08-heap.md`）。

---

## 规律总结

1. **排序是工具，先问「要不要排、按什么排」**：拿到题先判断无序状态下多做了什么比较；
   若排序后只需看相邻项或某个前缀，那一层排序就值。想清楚比较规则，代码往往只是一句
   `sort`。

2. **「有名次/位置」的题，排下标再回填**：506 是范本。排序键是值，但输出要按原下标，于是
   让下标跟着值一起排，用「第几名是谁」的名单把答案写回原位。

3. **自定义顺序翻译成「优先级 + 数值」的键**：1122、1636 都是把「先按某顺序、再按大小」
   写成二元组键，元组比较天然逐级进行。给「不在规则内」的元素一个统一的末尾优先级即可。

4. **自定义比较器要满足严格弱序**：179 用 `a+b > b+a` 当比较规则。若两个元素「不可分」
   （相等），比较器必须给出 0/相等判定，且不能在相等时返回「小于」，否则行为未定义。

5. **排序能帮我们锁定「最优解的形状」**：274 里最优 h 对应排序后的某个前缀；很多「至少 k
   个 / 最多 k 个」的题，排序后最优解都落在连续前缀或连续后缀上，扫描即可。

6. **种类少就划分，不必排序**：75 只有三种取值，直接三路划分做到 O(n) 且原地。元素种类
   为常数 k 时，三路划分（或计数排序）通常比比较排序更优。

7. **链表排序靠改指针，插入排序最自然**：147 用哨兵 + 逐个插入有序链。虚拟头结点消掉
   「插到头结点之前」的特例，与链表其它题（反转、删除、合并）是同一套边界消除术。

8. **要求线性时想桶，而不是比较排序**：比较排序下界是 O(n log n)，若值域/频率范围有限，
   计数或桶排序可到 O(n)。164 更进一步，靠**鸽巢原理**证明「答案只可能跨桶」，连桶内排序
   都省了。

9. **桶的两个常见坑**：桶宽别算成 0（整除结果要 `max(1, ...)`）、全等/单元素等退化情况要
   提前挡掉（除零）。另外别忘了处理**空桶**——最大间距可能跨过好几个空桶。

10. **稳定性与相等元素的处理**：让相等元素保持原有先后（稳定），要比较时用 `<=`/`<` 的
    选择来体现（147 的 `<=`、179 的严格 `>`）。是否需要稳定，取决于题目对同值次序有没有
    要求。

11. **「最小/最大」的初值要选对**：164 的 `prev` 初始化为 `lo`、作为「虚拟的前一个桶的
    最大值」，这样第一个非空桶也能参与比较。设计扫描型解法时，给「边界外」一个合理的哨兵
    初值，能省掉大量特判。
