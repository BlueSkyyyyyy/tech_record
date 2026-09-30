# 滑动窗口

滑动窗口解决的是这样一类问题：在**连续**的一段数组（或字符串）上求最优，而暴力枚举所有
区间的代价是 O(n²)。窗口的核心思想是——**右端只进不退，左端按需跟进**，让两个指针都单调右移，
于是每个元素最多进窗口一次、出窗口一次，整体降到 O(n)。

它有两个分支，搞清区别就抓住了主线：

- **定长窗**：窗口长度固定为 k，右端每走一步，左端就跟一步（如 643）；
- **变长窗**：窗口长度由条件决定，右端扩张，当条件被满足/破坏时再决定左端怎么动（如 209、3）。

变长窗里「左端怎么动」又分两种：一种是**一格一格收缩**（209，直到条件不再满足），
一种是**直接跳到该跳的位置**（3，跳到重复字符的下一位）。

当窗口要维护的对象从「和」换成「字符/元素的出现次数」时，就派生出**计数型窗口**：
用一张频次表（配一个「达标种类数」或「杂质数」）来判断窗口是否合法。这类题多、套路强，
是滑动窗口真正的重头戏（76、438、567、424、1004）。

此外还有两个特别的分支：**单调队列定长窗**（239，窗口求最值，队列里维护单调的下标）和
**乘积型变长窗**（713，把「和」换成「乘积」，并利用窗口一次性数出所有合法子数组）。
把这几类放在一起，滑动窗口的全貌才算完整。

本篇题目（由易到难）：

| 窗口类型 | 题目 | 难度 |
|---|---|---|
| 定长窗 | 643. 子数组最大平均数 I | 简单 |
| 变长窗（收缩） | 209. 长度最小的子数组 | 中等 |
| 变长窗（跳跃） | 3. 无重复字符的最长子串 | 中等 |
| 计数型变长窗（最短覆盖） | 76. 最小覆盖子串 | 困难 |
| 定长计数窗 | 438. 找到字符串中所有字母异位词 | 中等 |
| 定长计数窗 | 567. 字符串的排列 | 中等 |
| 可容忍杂质的最长窗 | 424. 替换后的最长重复字符 | 中等 |
| 可容忍杂质的最长窗 | 1004. 最大连续 1 的个数 III | 中等 |
| 单调队列定长窗 | 239. 滑动窗口最大值 | 困难 |
| 乘积型变长窗（计数） | 713. 乘积小于 K 的子数组 | 中等 |

---

## 模式一：定长滑动窗口

**适用信号**：题目明确固定了窗口长度（「长度为 k 的子数组」「连续 k 天」），求这个固定长度
窗口上的某个统计量的最优值。

核心动作：先建立前 k 个元素的窗口，之后每右移一格，就「加进一个新元素、吐出一个旧元素」，
用 O(1) 的代价更新统计量。由于窗口长度不变，左端完全被动地跟着右端走，不需要判断。

### 643. 子数组最大平均数 I（简单）

**题目**：给定整数数组 `nums` 和整数 `k`，找出长度为 `k` 的连续子数组的最大平均数。例如 `nums = [1, 12, -5, -6, 50, 3]`、`k = 4`，答案是 `[12, -5, -6, 50]`，平均数 `12.75`。

**思路（定长窗口维护和）**：
要求平均数最大，而长度 `k` 是固定的，所以「平均数最大」等价于「窗口和最大」——最后只差除以 `k` 这一步。
于是问题简化为：在所有长度为 `k` 的窗口里找最大的窗口和。

最直接的做法是对每个窗口重新求和，但相邻两个窗口之间有 `k - 1` 个元素是重合的，重复计算很浪费。
滑动窗口把它省掉：

1. 先求前 `k` 个元素之和，作为初始窗口；
2. 窗口整体右移一格时，只需 `加上新进来的 nums[i]、减去离开窗口的 nums[i - k]`；
3. 每次移动后用当前窗口和更新最大值；遍历结束后除以 `k`。

为什么这样是对的：窗口右移时，唯一的变化就是头尾两个元素，中间 `k - 2` 个原封不动。
既然要求的是整段的和，就没必要把它们重新加一遍。这个「增量更新」是定长窗的标志性写法。

**代码**（完整可运行版见 `src/sliding-window/maximum_average_subarray.py` / `.cpp`）：

```python
def find_max_average(nums, k):
    window = sum(nums[:k])
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]
        best = max(best, window)
    return best / k
```

```cpp
double findMaxAverage(const std::vector<int> &nums, int k) {
    long long window = 0;
    for (int i = 0; i < k; ++i) window += nums[i];
    long long best = window;
    for (int i = k; i < static_cast<int>(nums.size()); ++i) {
        window += nums[i] - nums[i - k];
        best = std::max(best, window);
    }
    return static_cast<double>(best) / k;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：窗口和可能很大，C++ 用 `long long` 更稳妥；除以 `k` 时先转 `double`，别写成整数除法；`k` 等于数组长度时循环不执行，返回初始窗口的平均数，这是对的。
- **相似题**：1456. 定长字符串中元音的最大数目、1343. 大小为 K 且平均值大于等于阈值的子数组数目（同为定长窗计数）；643 的变长版是 209（窗口长度不固定，求最小长度）。

---

## 模式二：变长滑动窗口（收缩左端）

**适用信号**：要找**满足某个和/计数条件**的最短（或最长）连续区间，且元素具有「越多越满足」
的单调性质（例如全是正数时，窗口越长、和越大）。

核心动作：右端一路扩张，一旦当前窗口满足条件，就在**保持条件满足的前提下尽量收缩左端**，
用一个 `while` 循环把左端往右推，每次收缩前先用当前窗口去更新答案。

### 209. 长度最小的子数组（中等）

**题目**：给定正整数数组 `nums` 和目标 `target`，找出**和大于等于 `target`** 的长度最小的连续子数组，返回其长度；不存在则返回 `0`。例如 `target = 7`、`nums = [2, 3, 1, 2, 4, 3]`，最短的是 `[4, 3]`，长度为 `2`。

**思路（右扩 + 满足条件就收缩）**：
右端 `right` 不断向右扩张，把 `nums[right]` 加进窗口和 `total`。此时看 `total`：

- 若 `total < target`，说明窗口还不够，继续扩右端；
- 若 `total >= target`，说明当前窗口已经满足条件。但题目要的是**最短**，所以尝试从左边吐出元素：
  先记录当前长度更新答案，再减去 `nums[left]` 并让 `left` 右移，重复到 `total < target` 为止。

为什么收缩是安全的：数组全是正数，从左边吐出一个元素，窗口和只会减小。所以只要 `total >= target`，
「把左端继续右移」得到的窗口都更短，是朝着最优解前进，不会错过任何更短的合法窗口。
反过来，如果数组里允许负数，「吐出一个元素反而可能让和更大」，收缩条件就乱了——这正是 560
必须用前缀和 + 哈希、而不能用滑动窗口的原因。

为什么整体是 O(n)：`right` 从 0 走到 n-1，每个元素只进窗口一次；`left` 也只会单调右移，每个元素
只出窗口一次。两层循环看似 O(n²)，实则两个指针各自走完全程，总步数 O(n)。这就是滑动窗口
「看似双层、实为线性」的经典论证。

**代码**（`src/sliding-window/minimum_size_subarray_sum.py` / `.cpp`）：

```python
def min_sub_array_len(target, nums):
    left = 0
    total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best
```

```cpp
int minSubArrayLen(int target, const std::vector<int> &nums) {
    int left = 0, total = 0, best = INT_MAX;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        total += nums[right];
        while (total >= target) {
            if (right - left + 1 < best) best = right - left + 1;
            total -= nums[left];
            ++left;
        }
    }
    return best == INT_MAX ? 0 : best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：更新答案必须写在**收缩之前**（此时窗口正好满足条件），否则会把已经不满足的窗口算进去；找不到解时返回 `0`，靠 `best` 是否仍为初始的无穷大/`INT_MAX` 判断；题目限定了**正整数**，这是能滑动的前提。
- **相似题**：3. 无重复字符的最长子串（同样变长窗，区别在左端怎么动）；76. 最小覆盖子串、438. 找到字符串中所有字母异位词（计数型变长窗，见本分类续篇）；560. 和为 K 的子数组（含负数，滑不动，改用哈希）。

---

## 模式三：变长滑动窗口（跳跃左端）

**适用信号**：要维护一个「窗口内元素互不冲突」的最长区间（典型是「无重复字符」），
一旦新元素与窗口内某个已有元素冲突，左端需要的不是一格一格挪，而是**直接跨过那个冲突位置**。

核心动作：用一个 `last[ch]` 表记录每个字符**上次出现的位置**。新字符进来时，如果它上次出现的位置
落在当前窗口内，就把左端跳到那个位置的下一位，窗口立刻恢复「无重复」。

### 3. 无重复字符的最长子串（中等）

**题目**：给定字符串 `s`，找出不含重复字符的**最长子串**的长度。例如 `s = "abcabcbb"`，最长无重复子串是 `"abc"`，长度为 `3`。

**思路（记录上次位置 + 左端跳跃）**：
仍然用 `[left, right]` 维护一个「无重复字符」的窗口，`right` 一路右扩。每遇到一个新字符 `ch`，
查它上次出现在 `last[ch]`：

- 如果 `last[ch] >= left`，说明这次出现的 `ch` 和窗口里那个旧的 `ch` 撞上了，当前窗口不再合法。
  此时把 `left` **直接跳到 `last[ch] + 1`**，把旧的 `ch` 甩出窗口；
- 否则说明窗口里没有 `ch`（或旧的 `ch` 早已在 `left` 左边、不构成冲突），`left` 不动。

然后更新 `last[ch] = right`，用 `right - left + 1` 更新答案。

为什么左端可以「跳」而不必一格一格挪：任何以「旧 `ch` 的位置及其左侧」为左端点、以当前
`right` 为右端点的窗口，都仍然包含两个 `ch`（当前的和旧的），都非法，而且只会比跳跃后的窗口更短。
所以这些左端可以直接一次性跳过，不会漏掉更优解。

为什么必须判断 `last[ch] >= left`：这是本题最容易写错的地方。反例 `s = "abba"`：处理到第二个 `b` 时
`left` 跳到 2；再遇到最后的 `a`，它上次出现在下标 0，但 `0 < left = 2`，说明那个 `a` 已经不在窗口里了，
不应该让 `left` 回退到 1。少了这个判断，答案会算错。

**代码**（`src/sliding-window/longest_substring_without_repeating.py` / `.cpp`）：

```python
def length_of_longest_substring(s):
    last = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

```cpp
int lengthOfLongestSubstring(const std::string &s) {
    std::unordered_map<char, int> last;
    int left = 0, best = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        char ch = s[right];
        auto it = last.find(ch);
        if (it != last.end() && it->second >= left) left = it->second + 1;
        last[ch] = right;
        best = std::max(best, right - left + 1);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(字符集大小)。
- **易错点**：`last[ch] >= left` 的判断不能省（见上面的 `"abba"`）；`left` 只能增大不能回退，所以更新 `last[ch] = right` 必须放在判断之后；空串返回 `0`。
- **相似题**：159 / 340. 至多包含 K 个不同字符的最长子串（把「无重复」放宽成「种类数不超过 K」，仍用 `last` 表配一个计数器）；424. 替换后的最长重复字符（窗口内「可容忍的杂质」计数）；76. 最小覆盖子串（变长窗的另一极：求最短的覆盖窗，见本分类续篇）。

---

## 模式四：计数型变长窗口（最短覆盖）

**适用信号**：要在一个字符串里找**最短**的、能覆盖另一个字符串全部字符（含重复次数）的窗口。
它和模式二（209）一样是「求最短的变长窗」，区别在于判断窗口是否合法的不是「和」，
而是一张**字符频次表**。

核心动作：用 `need` 记录目标串每个字符需要的次数，用 `window` 记录窗口里的实际次数，
再用一个整数 `formed` 记录「已经满足数量要求的字符种类数」。窗口合法当且仅当
`formed == len(need)`。合法时像 209 一样收缩左端，边收缩边更新最短答案。

### 76. 最小覆盖子串（困难）

**题目**：给定字符串 `s` 和 `t`，返回 `s` 中涵盖 `t` 所有字符（含重复次数）的最短子串；不存在则返回空串。例如 `s = "ADOBECODEBANC"`、`t = "ABC"`，答案是 `"BANC"`。

**思路（频次表 + 达标种类数）**：
这道题可以看成 209 的「字符串版」：209 判断的是窗口和是否达到 `target`，这里判断的是
窗口是否把 `t` 里每个字符都凑够了数量。做法：

1. `need` 统计 `t` 里每个字符要几个；
2. 右端 `right` 扩进来一个字符 `ch`，`window[ch]` 加一。如果 `ch` 是需要的、且刚好凑够
   `need[ch]`，就让 `formed` 加一；
3. 只要 `formed == len(need)`，说明当前窗口合法，就尝试收缩：先用当前长度更新最短答案，
   再吐出 `s[left]`；如果吐出后这个字符**不再达标**（`window < need`），`formed` 减一，
   窗口随之不再合法，停止收缩；
4. 回到步骤 2 继续扩右端。

为什么用 `formed`（达标字符**种类数**）而不是每次比较整张表：逐字符比较每次要 O(字符集)，
而 `formed` 只在某个字符的计数恰好跨过 `need` 阈值时增减，是 O(1) 的。这是本题从
「能过」到「优雅」的关键。注意阈值判断用 `==`（进窗口时）和 `<`（出窗口时），
因为「是否达标」是相对 `need` 的一次跨越，而不是单调过程。

为什么一旦合法就尽量收缩：越长越容易覆盖，合法窗口可能很长；我们要最短，所以只要还合法，
就把左端右移，直到不合法为止。这不会有遗漏——任何更短的合法窗口，都会在某一轮收缩中被枚举到。

**代码**（`src/sliding-window/minimum_window_substring.py` / `.cpp`）：

```python
def min_window(s, t):
    if not s or not t:
        return ""
    need = {}
    for ch in t:
        need[ch] = need.get(ch, 0) + 1
    window = {}
    formed = 0
    left = 0
    best_len = float("inf")
    best_left = 0
    for right, ch in enumerate(s):
        window[ch] = window.get(ch, 0) + 1
        if ch in need and window[ch] == need[ch]:
            formed += 1
        while formed == len(need):
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_left = left
            left_ch = s[left]
            window[left_ch] -= 1
            if left_ch in need and window[left_ch] < need[left_ch]:
                formed -= 1
            left += 1
    return "" if best_len == float("inf") else s[best_left:best_left + best_len]
```

```cpp
std::string minWindow(const std::string &s, const std::string &t) {
    if (s.empty() || t.empty()) return "";
    std::unordered_map<char, int> need, window;
    for (char c : t) need[c]++;
    int formed = 0;
    int left = 0, bestLen = INT_MAX, bestLeft = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        char c = s[right];
        window[c]++;
        if (need.count(c) && window[c] == need[c]) ++formed;
        while (formed == static_cast<int>(need.size())) {
            if (right - left + 1 < bestLen) {
                bestLen = right - left + 1;
                bestLeft = left;
            }
            char lc = s[left];
            window[lc]--;
            if (need.count(lc) && window[lc] < need[lc]) --formed;
            ++left;
        }
    }
    return bestLen == INT_MAX ? "" : s.substr(bestLeft, bestLen);
}
```

- **复杂度**：时间 O(|s| + |t|)，空间 O(字符集大小)。
- **易错点**：进窗口判 `==`、出窗口判 `<`，写成别的会少算或多算 `formed`；只有 `need` 里出现过的字符才能改动 `formed`，要加 `ch in need` 判断；找不到时返回空串，用 `best_len` 是否仍为无穷大判断；更新答案要写在**吐出元素之前**。
- **相似题**：209 长度最小的子数组（同一「最短」目标，判断量从频次表退化为和）；438 / 567（把「覆盖任意子串」收紧成「长度固定为 len(p)」，见下文）；713 乘积小于 K 的子数组（把「和」换成「乘积」，见模式八）。

---

## 模式五：定长计数窗口

**适用信号**：题目要求找「长度等于某个固定值、且字符组成与给定串一致」的子串，
典型是「字母异位词」和「排列」。窗口大小由目标串长度锁死，属于定长窗（模式一）的计数版本。

核心动作：开一张固定大小的计数表（如 26 个字母）。窗口右端每次进一个字符，一旦窗口长度
超过目标长度，就把最左边离开的字符计数减一，然后比较两张表是否相等。

### 438. 找到字符串中所有字母异位词（中等）

**题目**：给定字符串 `s` 和 `p`，找出 `s` 中所有是 `p` 的字母异位词的子串的起始下标。例如 `s = "cbaebabacd"`、`p = "abc"`，答案是 `[0, 6]`（`"cba"` 与 `"bac"`）。

**思路（固定窗口比较计数表）**：
字母异位词的长度一定等于 `p` 的长度，所以窗口大小固定为 `len(p)`。我们只需枚举每个长度为
`len(p)` 的窗口，看它各字母计数是否与 `p` 相同：

1. `need` 记录 `p` 的字母计数，`window` 记录当前窗口的字母计数；
2. 右端 `i` 进来一个字符，`window` 加一；若窗口长度超过 `len(p)`，把 `s[i - len(p)]`
   的计数减一（它正好滑出了窗口）；
3. 当窗口已满（`i >= len(p) - 1`）且 `window == need`，把窗口左端下标 `i - len(p) + 1` 记下。

为什么用定长窗：判定条件本身含「长度必须等于 `len(p)`」这一硬约束，右端每走一步左端被动跟一步，
不需要判断何时收缩，正是定长窗。为什么比较整张表就对：异位词只在乎每个字母出现几次，
不在乎顺序；两表逐项相等，就说明组成一致。把字母表固定成 26，比较是常数时间。

**代码**（`src/sliding-window/find_all_anagrams.py` / `.cpp`）：

```python
def find_anagrams(s, p):
    res = []
    n, m = len(s), len(p)
    if n < m:
        return res
    need = [0] * 26
    window = [0] * 26
    for ch in p:
        need[ord(ch) - ord("a")] += 1
    for i, ch in enumerate(s):
        window[ord(ch) - ord("a")] += 1
        if i >= m:
            window[ord(s[i - m]) - ord("a")] -= 1
        if i >= m - 1 and window == need:
            res.append(i - m + 1)
    return res
```

```cpp
std::vector<int> findAnagrams(const std::string &s, const std::string &p) {
    std::vector<int> res;
    int n = static_cast<int>(s.size()), m = static_cast<int>(p.size());
    if (n < m) return res;
    std::array<int, 26> need{}, window{};
    for (char c : p) need[c - 'a']++;
    for (int i = 0; i < n; ++i) {
        window[s[i] - 'a']++;
        if (i >= m) window[s[i - m] - 'a']--;
        if (i >= m - 1 && window == need) res.push_back(i - m + 1);
    }
    return res;
}
```

- **复杂度**：时间 O(|s|)，空间 O(1)（字母表固定 26）。
- **易错点**：`n < m` 要先特判，否则窗口永远不满、也不会报错；「进窗口」与「出窗口」的顺序不能反，先加新字符再删旧字符；判断 `i >= m - 1` 才能读窗口数据；C++ 里 `std::array` 支持 `==` 比较，非常方便，别自己写循环。
- **相似题**：567 字符串的排列（同一个模型，命中即返回，见下）；76 最小覆盖子串（去掉「长度固定」的约束，就退化成模式四）；242 有效的字母异位词（只判断一次，是本题的退化版，见 `02-hash`）。

### 567. 字符串的排列（中等）

**题目**：给定字符串 `s1` 和 `s2`，判断 `s2` 中是否包含 `s1` 的某个排列（长度相同、字母组成相同的连续子串）。例如 `s1 = "ab"`、`s2 = "eidbaooo"`，答案是 `true`（`"ba"` 是一个排列）。

**思路（定长计数窗，命中即返回）**：
`s1` 的排列就是与 `s1` 字母组成一致的连续子串，窗口长度固定为 `len(s1)`。用两本 26 长的
计数表分别表示「`s1` 需要」和「当前窗口」，右端进一个、窗口超长时左端出一个，两表相等
即找到排列。

本题与 438 是同一个模型，唯一区别是：438 要把**所有**匹配位置收集起来，本题找到一个就能
提前返回，所以把「收集下标」换成「命中即真」，其余逐字相同。灵活一点看，**会做 438 就必然会做 567**，
面试遇到时可以主动点破这层关系。

**代码**（`src/sliding-window/permutation_in_string.py` / `.cpp`）：

```python
def check_inclusion(s1, s2):
    n1, n2 = len(s1), len(s2)
    if n1 > n2:
        return False
    need = [0] * 26
    window = [0] * 26
    for ch in s1:
        need[ord(ch) - ord("a")] += 1
    for i, ch in enumerate(s2):
        window[ord(ch) - ord("a")] += 1
        if i >= n1:
            window[ord(s2[i - n1]) - ord("a")] -= 1
        if i >= n1 - 1 and window == need:
            return True
    return False
```

```cpp
bool checkInclusion(const std::string &s1, const std::string &s2) {
    int n1 = static_cast<int>(s1.size()), n2 = static_cast<int>(s2.size());
    if (n1 > n2) return false;
    std::array<int, 26> need{}, window{};
    for (char c : s1) need[c - 'a']++;
    for (int i = 0; i < n2; ++i) {
        window[s2[i] - 'a']++;
        if (i >= n1) window[s2[i - n1] - 'a']--;
        if (i >= n1 - 1 && window == need) return true;
    }
    return false;
}
```

- **复杂度**：时间 O(|s2|)，空间 O(1)（字母表固定 26）。
- **易错点**：`s1` 比 `s2` 长时直接 `false`；注意本题 `s1` 是「模式」、`s2` 是「文本」，别把两串写反；其余同 438。
- **相似题**：438 找到字符串中所有字母异位词（同一个模型的「收集全部」版本）；76 最小覆盖子串（约束放宽后变成求最短覆盖）；面试里常被追问「如果 `s1` 里有重复字符怎么办」——两本计数表天然支持重复，无需额外处理。

---

## 模式六：可容忍杂质的最长窗口

**适用信号**：求一段最长区间，允许区间里存在**不超过 k 个」的「杂质」，把杂质改成目标字符后整段就统一。
典型说法是「最多替换 k 个字符」「最多翻转 k 个 0」。

核心动作：维护窗口内「杂质」的计数。设窗口内出现次数最多的字符有 `max_freq` 个，那么需要改动的
字符数就是「窗口长度 − `max_freq`」。只要它不超过 k，窗口就合法；超过就收缩左端。

### 424. 替换后的最长重复字符（中等）

**题目**：给定只含大写字母的字符串 `s` 和整数 `k`，最多替换 `k` 个字符，求能得到的最长重复字符子串的长度。例如 `s = "AABABBA"`、`k = 1`，答案是 `4`（把中间的 `B` 换成 `A` 得 `"AAAA"`）。

**思路（窗口长度 − 最多频次 ≤ k）**：
要让窗口内所有字符都变成同一个字符，改动次数 = 窗口长度 − 窗口内出现次数最多的字符的频次。
例如窗口是 `"AABA"`，`A` 有 3 个、`B` 有 1 个，只需把 `B` 改成 `A`，改动 1 次。

于是右端一路右扩，维护各字符计数以及历史最大值 `max_freq`。当
`(right - left + 1) - max_freq > k` 时，把左端右移一格。

为什么这里用 `if` 而不用 `while`：窗口长度最多比上一轮大 1，所以条件一旦被破坏，左端
右移一格必然能恢复。既然要的是「最长」，窗口长度从头到尾只增不减，用当前长度和历史最大
取大即可。用 `while` 也行，只是多做无用功。

为什么 `max_freq` 缩小后不用回退：`max_freq` 记的是**历史**最大频次。它对应的字符可能已经
滑出窗口，于是这个值偏大；但偏大只会让 `(窗口长度 - max_freq)` 偏小、条件更宽松。我们要的
是最大长度，偏宽松不会给出「更大却非法」的答案——任何被判为合法的窗口，其真实改动次数只会
不超过这个估计值。这是在草稿纸上走一遍 `"AABABBA"` 就能验证的巧妙之处。

**代码**（`src/sliding-window/longest_repeating_character_replacement.py` / `.cpp`）：

```python
def character_replacement(s, k):
    count = {}
    left = 0
    best = 0
    max_freq = 0
    for right, ch in enumerate(s):
        count[ch] = count.get(ch, 0) + 1
        max_freq = max(max_freq, count[ch])
        if (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

```cpp
int characterReplacement(const std::string &s, int k) {
    std::array<int, 26> count{};
    int left = 0, best = 0, maxFreq = 0;
    for (int right = 0; right < static_cast<int>(s.size()); ++right) {
        int idx = s[right] - 'A';
        ++count[idx];
        if (count[idx] > maxFreq) maxFreq = count[idx];
        if ((right - left + 1) - maxFreq > k) {
            --count[s[left] - 'A'];
            ++left;
        }
        if (right - left + 1 > best) best = right - left + 1;
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)（字母表固定 26）。
- **易错点**：`max_freq` 是历史最大值，不要在看到它变小后去回退更新；收缩用 `if` 即可（每轮至多缩一格），若改用 `while` 也要保证每次收缩只减一次左端字符计数；答案在「收缩后」用当前窗口长度更新，因为不合法时窗口长度恰好保持上一轮的值，不影响正确性。
- **相似题**：1004 最大连续 1 的个数 III（同模型，杂质固定为 0，见下）；3 无重复字符的最长子串（「杂质」是重复字符，阈值是「一个都不能有」）；1493 删掉一个元素以后全为 1 的最长子数组（k = 1 的特例）。

### 1004. 最大连续 1 的个数 III（中等）

**题目**：给定二进制数组 `nums` 和整数 `k`，最多可以把 `k` 个 `0` 翻成 `1`，求能得到的连续 `1` 的最大个数。例如 `nums = [1,1,1,0,0,0,1,1,1,1,0]`、`k = 2`，答案是 `6`。

**思路（杂质固定为 0）**：
这题和 424 是同一个模型，只是「杂质」固定为 `0`。窗口内有多少个 `0`，就需要翻转多少次；
只要 `zeros <= k`，窗口就合法，窗口长度就是一段连续 `1`。

右端右扩，遇到 `0` 就把 `zeros` 加一；当 `zeros > k` 时收缩左端，若离开的是 `0` 则 `zeros` 减一，
直到窗口重新合法。用窗口长度更新答案。

两题的映射关系：424 的「窗口长度 − `max_freq`」就是「需要改动的字符数」，本题的
「`zeros`」也是「需要改动的字符数」。区别只是 424 要在窗口里现算最多频次，而本题目标字符
固定为 `1`，杂质数直接数 `0` 即可。会做 424 就直接会做 1004。

**代码**（`src/sliding-window/max_consecutive_ones_iii.py` / `.cpp`）：

```python
def longest_ones(nums, k):
    left = 0
    zeros = 0
    best = 0
    for right, x in enumerate(nums):
        if x == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

```cpp
int longestOnes(const std::vector<int> &nums, int k) {
    int left = 0, zeros = 0, best = 0;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        if (nums[right] == 0) ++zeros;
        while (zeros > k) {
            if (nums[left] == 0) --zeros;
            ++left;
        }
        if (right - left + 1 > best) best = right - left + 1;
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：收缩左端时**只有离开的是 `0`** 才让 `zeros` 减一，别无条件减；`while` 与 `if` 在这里都可以（每轮至多缩一格），但语义上「直到合法」用 `while` 更直白；`k = 0` 时退化求最长连续 `1`，验证一下 `[1,1,1]` 应得 3。
- **相似题**：424 替换后的最长重复字符（一般化版本）；485 最大连续 1 的个数（`k = 0` 的特例）；487 最大连续 1 的个数 II（`k = 1` 的特例）；1493 删掉一个元素以后全为 1 的最长子数组。

---

## 模式七：单调队列定长窗口（求最值）

**适用信号**：窗口长度固定，但要维护的是窗口内的**最值**（最大/最小），而不是和或计数。
求和的定长窗靠「加减增量」即可，但最大值不能这样增量维护（离开的如果是最大值，增量法就失效了）。

核心动作：用一个双端队列 `deque` 存**下标**，保持对应值从队首到队尾单调递减。
新元素进来时从队尾弹掉所有比它小的；队首超出窗口就弹出。队首始终是当前窗口最大值。

### 239. 滑动窗口最大值（困难）

**题目**：给定整数数组 `nums` 和窗口大小 `k`，窗口从最左滑到最右，返回每个窗口内的最大值组成的数组。例如 `nums = [1,3,-1,-3,5,3,6,7]`、`k = 3`，答案是 `[3,3,5,5,6,7]`。

**思路（单调递减队列）**：
窗口右端进来一个 `x` 时，先把队尾所有 **值 ≤ x** 的下标弹掉——它们不可能再成为任何窗口的
最大值：只要 `x` 还在窗口里，这些旧值就一定被 `x` 压住；而 `x` 失效得比它们晚（`x` 更靠右，
活得更久），等 `x` 滑出时它们早就滑出了。然后把当前下标入队。
接着检查队首是否滑出窗口（下标 ≤ `i - k`），是就弹出。此时队首就是当前窗口最大值的下标。

为什么队列存的是下标而不是值：需要判断队首是否已滑出窗口，只有下标能提供这个信息。

为什么整体是 O(n)：每个下标最多入队一次；一旦被弹出就再也不会回来。内层虽有 `while`，
但它执行的次数不超过总入队次数，所以是摊还 O(n)。这和模式二的「看似双层、实为线性」是
同一类论证，只是这里靠的是「每个下标进出队各一次」。

**代码**（`src/sliding-window/sliding_window_maximum.py` / `.cpp`）：

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

```cpp
std::vector<int> maxSlidingWindow(const std::vector<int> &nums, int k) {
    std::deque<int> dq;  // 存下标，对应值单调递减
    std::vector<int> res;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() <= i - k) dq.pop_front();
        if (i >= k - 1) res.push_back(nums[dq.front()]);
    }
    return res;
}
```

- **复杂度**：时间 O(n)，空间 O(k)。
- **易错点**：队列里存下标、比较时取 `nums[...]` 的值，别混用；弹队首要判「是否已滑出」用 `<= i - k`（窗口是 `[i-k+1, i]`）；弹队尾用 `<=` 而非 `<`，两者都对，但 `<=` 能让队列保持「新值优先」，实现更简单；只有窗口形成后才往结果里写（`i >= k - 1`）。
- **相似题**：155 最小栈（也是「随时取最值」的结构，见 `stack`）；剑指 Offer 59-I 滑动窗口的最大值（同题）；补充：若把「最大」换成「最小」，把队尾比较方向反过来即可。

---

## 模式八：乘积型变长窗口（计数）

**适用信号**：把模式二里的「和」换成「乘积」，统计满足「乘积 < k」的**子数组个数**（而不是最短长度）。
元素都是正数，所以窗口越长乘积越大，单调性依然成立。

核心动作：维护一个乘积 `prod < k` 的窗口，右端扩张、超限就收缩左端。计数时利用一个性质：
窗口 `[left, right]` 合法时，以 `right` 结尾的合法子数组恰有 `right - left + 1` 个，
一次性把它们全加上。

### 713. 乘积小于 K 的子数组（中等）

**题目**：给定正整数数组 `nums` 和整数 `k`，返回乘积严格小于 `k` 的连续子数组的个数。例如 `nums = [10,5,2,6]`、`k = 100`，答案是 `8`。

**思路（窗口合法时一次性计数）**：
窗口 `[left, right]` 表示当前「乘积 < k」的区间。右端进来 `x` 后如果乘积 `>= k`，就从左边
不断吐出元素，直到乘积再次 `< k`。

关键在于**怎么数**：当窗口合法时，以 `right` 结尾的合法子数组有 `right - left + 1` 个。
理由是——左端可以取 `left, left+1, …, right` 中任意一个位置；左端越靠右，区间越短，
乘积也越小，而当前 `[left, right]` 已经合法，所以这些更短的区间必然都合法。于是把
`right - left + 1` 累加即可，根本不用枚举子数组。

为什么要特判 `k <= 1`：元素都是正整数，任何子数组的乘积至少是 1，不可能严格小于 `k`，
直接返回 0（同时避免了 `k = 0` 时收缩循环除到 0 的边界问题）。

为什么可以放心收缩：全是正数，吐出一个元素乘积只会变小或不变；`prod < k` 一成立就停，
此时左端最靠右、合法子数组数最多，`right - left + 1` 正是以 `right` 结尾的全部合法区间。

**代码**（`src/sliding-window/subarray_product_less_than_k.py` / `.cpp`）：

```python
def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0
    prod = 1
    left = 0
    count = 0
    for right, x in enumerate(nums):
        prod *= x
        while prod >= k:
            prod //= nums[left]
            left += 1
        count += right - left + 1
    return count
```

```cpp
int numSubarrayProductLessThanK(const std::vector<int> &nums, int k) {
    if (k <= 1) return 0;
    long long prod = 1;
    int left = 0, count = 0;
    for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
        prod *= nums[right];
        while (prod >= k) {
            prod /= nums[left];
            ++left;
        }
        count += right - left + 1;
    }
    return count;
}
```

- **复杂度**：时间 O(n)，空间 O(1)。
- **易错点**：`k <= 1` 必须特判，否则 `k = 0` 时 `while` 会把左端推到右端之外、甚至除零；计数是 `right - left + 1`（以 `right` 结尾的个数），不是「窗口长度」的简单重复；C++ 用 `long long` 存乘积防溢出。
- **相似题**：209 长度最小的子数组（同一窗口形状，求的是最短长度而非个数）；560 和为 K 的子数组（把「乘积 < k」换成「和 = k」，由于存在负数需改用前缀和 + 哈希，见 `02-hash`）；713 与 992（K 个不同整数的子数组）都用到「以 `right` 结尾计数」这一技巧。

---

## 规律总结

1. **先分清定长还是变长**。题目给了窗口长度 k 就是定长窗，右端走一步左端跟一步（643）；没说固定长度、要自己找最优区间就是变长窗（209、3）。这是拿到题的第一判断。
2. **定长窗的记忆点是「增量更新」**：右移时只加新元素、只减旧元素，绝不要重新算整段。凡是「相邻窗口高度重叠」的场景，都能这样省掉重复计算。
3. **变长窗的记忆点是「右端只进不退」**：`right` 用一个 `for` 从头走到尾，`left` 在一个 `while` 里按需右移。两个指针都单调，所以尽管有两层循环，复杂度仍是 O(n)（209、3 都靠这句证明）。
4. **收缩时机的区别**：209 在「满足条件」时收缩，用 `while` 一格一格吐，求最短；3 在「出现冲突」时直接跳到冲突位置的下一位，求最长。前者是「缩到不再满足」，后者是「一步到位消除冲突」。
5. **滑动窗口成立的前提是「单调性」**：窗口越长，某个统计量只增不减（或只减不增），收缩才有明确方向。209 要求正整数、3 要求「重复即非法」，都是这个单调性。一旦数组可含负数（560），单调性被破坏，就该换前缀和 + 哈希。
6. **左端移动前先更新答案**：209 里「`total >= target` 的那一刻窗口才合法」，所以答案更新要写在吐出元素之前。顺序写反，统计到的是收缩后的窗口，会偏小。
7. **「看似双层、实为线性」要靠指针单调性论证**：写题解时若声称 O(n) 却有两层循环，一定要说清「每个元素最多进出窗口各一次」。这是滑动窗口区别于暴力枚举的关键，也是面试最想听的解释。239 的单调队列同理，靠「每个下标进出队各一次」。
8. **遇到计数型窗口就开频次表**：窗口判断量从「和」变成「字符出现次数」时，用一张计数表（字符集固定就开定长数组，否则用哈希表）。判断窗口是否合法有两种写法：比较整张表（438、567，适合作业式全部收集），或用「达标种类数」（76）／「杂质数」（424、1004）这样的**单个整数指标**，后者把每次比较降到 O(1)，是更通用的高效写法。
9. **「窗口长度 ≥ 某个量」要用历史最大值**：424 里 `max_freq` 只增不减，原因是要的是最长窗口、且偏宽松不会给出非法的「更大」答案。凡是「求最长 + 指标可能回退」的场景，都要想一想能不能用历史值替代实时值。
10. **求窗口最值用单调队列，别用增量**：和可以增量更新，最值不行（离开的恰好是最大值时就崩了）。239 用存下标的单调递减队列表述「谁在我之后才失效」，是求最值窗口的标准工具。
11. **四类窗口的分工**：定长统计（643、438、567）、变长求最短（209、76）、变长求最长（3、424、1004）、定长求最值（239）、变长计数（713）。拿到题先对号入座，再决定「左端是否被动跟」「何时收缩」「判断量是什么」。
12. **相似题就是换一个判断量**：209 → 76 是把和换成频次表，424 → 1004 是把不舒服的字符换成 0，209 → 713 是把长度换成个数。滑动窗口的题目彼此之间往往只差一个「合法条件」的定义，抓住这个定义，就能把一整组题串起来。
