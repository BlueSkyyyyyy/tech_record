# 哈希表

哈希表（散列表）是刷题里最「作弊」也最实用的工具：它把「查找某个值」从 O(n) 压到接近 O(1)，
于是很多需要双层循环的题，一次线性扫描就能解决。它的代价是额外的空间——这就是经典的**空间换时间**。

本篇不讲哈希表的底层实现，只讲它在题目里最常见的五组用法：
**边扫边记**（用「我在等谁」代替「谁和我配」）、**计数与去重**、**双向映射**、
**指纹分组**、以及**集合判定（判环与起点扫描）**。这五组覆盖了哈希类题目的绝大多数场景。

本篇题目（由易到难）：

| 用法 | 题目 | 难度 |
|---|---|---|
| 边扫边记 | 1. 两数之和 | 简单 |
| 计数与抵消 | 242. 有效的字母异位词 | 简单 |
| 集合去重 | 349. 两个数组的交集 | 简单 |
| 计数与抵消 | 383. 赎金信 | 简单 |
| 双向映射 | 205. 同构字符串 | 简单 |
| 集合判环 | 202. 快乐数 | 简单 |
| 指纹分组 | 49. 字母异位词分组 | 中等 |
| 边扫边记（分治） | 454. 四数相加 II | 中等 |
| 边扫边记（前缀和） | 560. 和为 K 的子数组 | 中等 |
| 集合判定 + 起点扫描 | 128. 最长连续序列 | 中等 |

---

## 模式一：边扫边记（用「欠账」代替「配对」）

适用信号：要在一个**无序**数组里找满足某种关系的两个（或多个）元素，且这种关系可以改写成
「我需要一个什么样的值」。最典型的就是两数之和。

### 1. 两数之和（简单）

**题目**：给定整数数组 `nums` 和目标值 `target`，找出和为 `target` 的两个数，返回它们的下标。恰有一个答案，同一元素不能用两次。例如 `nums = [2, 7, 11, 15]`、`target = 9`，返回 `[0, 1]`。

**思路（一次扫描 + 哈希表）**：
最容易想到的是两层循环枚举所有数对，但那是 O(n²)。换一个视角：

> 遍历到 `x` 时，与其去猜「谁能和 `x` 配成 `target`」，不如反过来问「`x` 在等谁来配对」。

`x` 需要的就是 `target - x`。如果这个值之前出现过，那它一定和当前的 `x` 配成一对，答案就是
「它当时的下标」和「当前下标」。如果还没出现，就把 `x` 和它的下标记下来，让后面的数来找它。

为什么这样不会用同一个元素两次：代码是**先查后存**。遍历到 `x` 时，表里只有它**之前**的元素，
`x` 自己还没进去，所以不可能出现「自己和自己配对」。

这也解释了为什么无序数组要用哈希而不能用对撞双指针：对撞双指针依赖「有序」来判断该缩哪一端，
而哈希表不依赖顺序，它把「顺序信息」换成了「值到下标的映射」。

**代码**（完整可运行版见 `src/hash/two_sum.py` / `.cpp`）：

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []
```

```cpp
std::vector<int> twoSum(const std::vector<int> &nums, int target) {
    std::unordered_map<int, int> seen;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        int need = target - nums[i];
        auto it = seen.find(need);
        if (it != seen.end()) return {it->second, i};
        seen[nums[i]] = i;
    }
    return {};
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：一定要**先查后存**，否则 `[3, 3]`、`target = 6` 这种会把自己算进去；存的是「值 → 下标」，若数组有重复值，后出现的会覆盖先出现的，但因为我们是一查到就立刻返回，被覆盖的那个值已经完成使命，所以不影响结果。
- **相似题**：167. 两数之和 II（**有序**数组，用对撞双指针，见数组篇）；653. 两数之和 IV（BST 上做同样的「边扫边记」）；454. 四数相加 II（把两两之和先记进哈希表）；560. 和为 K 的子数组（把前缀和记进哈希表）。

### 454. 四数相加 II（中等）

**题目**：给定四个等长整数数组 `A`、`B`、`C`、`D`，统计有多少个四元组 `(i, j, k, l)` 满足 `A[i] + B[j] + C[k] + D[l] == 0`。

**思路（两两之和 + 哈希计数）**：
四层循环枚举所有四元组是 O(n⁴)，太慢。注意题目**不要求去重、只要求统计方案数**，
这正是哈希表最擅长的。把四个数组分成两半，各算「两两之和」：

1. 枚举 `A`、`B` 的所有组合，把和以及它出现的**次数**记进哈希表；
2. 枚举 `C`、`D` 的所有组合，和为 `s` 时，查表里有多少个 `-s`，这些组合都能和当前 `(k, l)` 配成 0。

一次「两两枚举」是 O(n²)，哈希表把两半接起来，总复杂度 O(n²)。

为什么存次数而不是只存「和是否出现」：不同的 `(i, j)` 即使和相同，也是不同方案。
比如 `A = [1, 2]`、`B = [-2, -1]`，`(0,0)` 与 `(1,1)` 的和都是 `-1`，必须都数上。

**代码**（`src/hash/four_sum_ii.py` / `.cpp`）：

```python
def four_sum_count(a, b, c, d):
    ab = {}
    for x in a:
        for y in b:
            ab[x + y] = ab.get(x + y, 0) + 1
    total = 0
    for x in c:
        for y in d:
            total += ab.get(-(x + y), 0)
    return total
```

```cpp
int fourSumCount(const std::vector<int> &a, const std::vector<int> &b,
                 const std::vector<int> &c, const std::vector<int> &d) {
    std::unordered_map<int, int> ab;
    for (int x : a)
        for (int y : b) ++ab[x + y];
    int total = 0;
    for (int x : c)
        for (int y : d) {
            auto it = ab.find(-(x + y));
            if (it != ab.end()) total += it->second;
        }
    return total;
}
```

- **复杂度**：时间 O(n²)，空间 O(n²)。
- **易错点**：统计的是**方案数**不是「是否存在」，哈希表存的是计数；`-s` 别写成 `s`。
- **相似题**：1. 两数之和（两数版，边扫边记）；18. 四数之和（要求结果不重复，用排序 + 双指针，反而不用哈希）；560. 和为 K 的子数组（同样是「记录中间和再查目标」）。

### 560. 和为 K 的子数组（中等）

**题目**：给定整数数组 `nums` 和整数 `k`，统计有多少个**连续子数组**的元素和恰好等于 `k`。

**思路（前缀和 + 哈希计数）**：
暴力枚举所有子数组求和是 O(n²)。用前缀和的视角加速。记 `prefix[i]` 为「前 i 个元素之和」，
则子数组 `nums[j..i-1]` 的和等于 `prefix[i] - prefix[j]`。要求它等于 `k`，即：

> `prefix[j] == prefix[i] - k`

于是边扫描边维护一张哈希表：`前缀和 -> 它出现过的次数`。扫描到位置 `i` 时，
把「表里 `prefix[i] - k` 出现的次数」累加进答案，就是以 `i` 结尾的合法子数组个数。

为什么初始化 `count[0] = 1`：它代表**空前缀**，对应 `j = 0` 的情况。
没有它，所有「从下标 0 开始」的子数组都会被漏掉。

为什么不能像「长度最小的子数组」那样用滑动窗口：数组可能含负数，前缀和不再单调递增，
窗口该缩还是该扩失去了判断依据。哈希表则不受单调性限制。

**代码**（`src/hash/subarray_sum_equals_k.py` / `.cpp`）：

```python
def subarray_sum(nums, k):
    count = {0: 1}
    prefix = 0
    total = 0
    for x in nums:
        prefix += x
        total += count.get(prefix - k, 0)
        count[prefix] = count.get(prefix, 0) + 1
    return total
```

```cpp
int subarraySum(const std::vector<int> &nums, int k) {
    std::unordered_map<int, int> count{{0, 1}};
    int prefix = 0, total = 0;
    for (int x : nums) {
        prefix += x;
        auto it = count.find(prefix - k);
        if (it != count.end()) total += it->second;
        ++count[prefix];
    }
    return total;
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：`count[0] = 1` 的初始化和「先查后存」的顺序都不能少；注意统计的是**前缀和出现次数**，重复前缀和要累加（`[1, -1, 0]`、`k = 0` 有 3 个子数组，就是重复前缀和的功劳）。
- **相似题**：1. 两数之和（把「目标和」换成「值相等」）；454. 四数相加 II（把数组劈成两半，记录一半的和）；974. 和可被 K 整除的子数组（把前缀和对 K 取模再计数，见前缀和篇）。

---

## 模式二：计数与去重

适用信号：题目关心的是「某个元素出现没出现 / 出现几次」，而不是它的位置。此时把数组
「压扁」成一个集合（去重）或一张计数字典（频次），问题往往就退化成对集合的简单判断。

### 242. 有效的字母异位词（简单）

**题目**：给定字符串 `s` 和 `t`，判断 `t` 是否是 `s` 的字母异位词——即每个字符出现的次数都相同，只是排列顺序不同。例如 `s = "anagram"`、`t = "nagaram"` 是异位词。

**思路（字符计数）**：
「异位词」的本质就是「字符频次完全一致」，与顺序无关。所以先统计 `s` 的字符频次，再遍历 `t`
逐一**抵消**：

- 遍历 `t` 时若某个字符的计数已经是 0，说明 `t` 多用了这个字符，直接判否；
- 全部抵消完还没出问题，就是异位词。

先比较长度是个便宜的前置剪枝：长度不同必然不是异位词。

因为题目限定小写字母，其实用长度 26 的数组装计数也行；哈希表的写法对任意字符集都成立，更通用。

**代码**（`src/hash/valid_anagram.py` / `.cpp`）：

```python
def is_anagram(s, t):
    if len(s) != len(t):
        return False
    count = {}
    for ch in s:
        count[ch] = count.get(ch, 0) + 1
    for ch in t:
        if count.get(ch, 0) == 0:
            return False
        count[ch] -= 1
    return True
```

```cpp
bool isAnagram(const std::string &s, const std::string &t) {
    if (s.size() != t.size()) return false;
    std::unordered_map<char, int> count;
    for (char ch : s) ++count[ch];
    for (char ch : t) {
        if (count[ch] == 0) return false;
        --count[ch];
    }
    return true;
}
```

- **复杂度**：时间 O(n)，空间 O(1)（小写字母只有 26 种，表规模有上界）。
- **易错点**：`count.get(ch, 0)` 或 C++ 里 `count[ch]` 在键不存在时默认是 0，判断「计数是否为 0」即可，不必额外判 key 存在；先比长度能省掉后面的工作。
- **相似题**：49. 字母异位词分组（把同样的计数思想扩展到多串分组的「指纹」）；383. 赎金信（用 `t` 的字符去拼 `s`，是同一套抵消逻辑）；49 与本题可对照看：一个是「判断一组」，一个是「把多组分开」。

### 349. 两个数组的交集（简单）

**题目**：给定两个数组 `nums1` 和 `nums2`，返回它们的交集。结果中每个元素唯一，顺序不限。例如 `nums1 = [4, 9, 5]`、`nums2 = [9, 4, 9, 8, 4]`，返回 `[9, 4]`（顺序不限）。

**思路（集合 + 边收集边删除）**：
题目要的是「出现在两个数组里的值」，不关心次数。所以：

1. 先把 `nums1` 装进一个集合 `set1`，完成去重；
2. 遍历 `nums2`，遇到在 `set1` 里的元素就收进结果，**同时把它从 `set1` 删掉**。

为什么删掉：`nums2` 里可能有重复值（比如上面的两个 9），若不删除就会重复收进结果。删掉之后
同一个值最多被收一次，天然满足「结果元素唯一」。

**代码**（`src/hash/intersection_of_two_arrays.py` / `.cpp`）：

```python
def intersection(nums1, nums2):
    set1 = set(nums1)
    res = []
    for x in nums2:
        if x in set1:
            res.append(x)
            set1.discard(x)
    return res
```

```cpp
std::vector<int> intersection(const std::vector<int> &nums1,
                              const std::vector<int> &nums2) {
    std::unordered_set<int> set1(nums1.begin(), nums1.end());
    std::vector<int> res;
    for (int x : nums2) {
        auto it = set1.find(x);
        if (it != set1.end()) {
            res.push_back(x);
            set1.erase(it);
        }
    }
    return res;
}
```

- **复杂度**：时间 O(m + n)，空间 O(m)（或把较短数组装进集合，空间 O(min(m, n))）。
- **易错点**：C++ 里 `set1.erase(it)` 用迭代器版本，别先 `erase(x)` 又用 `it`（迭代器会失效）；结果顺序取决于遍历顺序，题目不要求有序，判题时不能假定顺序。
- **相似题**：350. 两个数组的交集 II（改成保留出现次数，要用计数哈希表而不是集合）；349 的反向是「差集」，思路同为集合运算。

### 383. 赎金信（简单）

**题目**：给定两个字符串 `ransomNote` 和 `magazine`，判断 `ransomNote` 能否由 `magazine` 中的字符拼成。`magazine` 里每个字符只能使用一次。例如 `ransomNote = "aa"`、`magazine = "aab"` 可以，而 `magazine = "ab"` 不行。

**思路（字符计数 + 逐个扣减）**：
「能不能拼成」等价于「`ransomNote` 里每个字符的出现次数都不超过 `magazine`」。
这就是上一题「计数与抵消」的翻版：

1. 先统计 `magazine` 的字符频次；
2. 遍历 `ransomNote` 逐个扣减，若某个字符计数已经为 0，说明供不上，返回 `False`；
3. 全部扣完没出问题，就能拼成。

与 242 的区别：242 要求两个字符串**完全等量**（互相抵消到零），而本题只要求 `magazine`
「够用」即可，是一个**子集/资源约束**判定。计数表恰好同时表达这两种关系，区别只在最后怎么判断。

**代码**（`src/hash/ransom_note.py` / `.cpp`）：

```python
def can_construct(ransom_note, magazine):
    count = {}
    for ch in magazine:
        count[ch] = count.get(ch, 0) + 1
    for ch in ransom_note:
        if count.get(ch, 0) == 0:
            return False
        count[ch] -= 1
    return True
```

```cpp
bool canConstruct(const std::string &ransomNote, const std::string &magazine) {
    std::unordered_map<char, int> count;
    for (char ch : magazine) ++count[ch];
    for (char ch : ransomNote) {
        if (count[ch] == 0) return false;
        --count[ch];
    }
    return true;
}
```

- **复杂度**：时间 O(m + n)，空间 O(字符集大小)。
- **易错点**：不要先判断长度（`ransomNote` 比 `magazine` 长必然不行，但短不一定行，长度不是关键条件）；`count[ch] == 0` 在键不存在时也为真，无需额外判 key。
- **相似题**：242. 有效的字母异位词（等量版）；49. 字母异位词分组（把计数推广成指纹）；本题是「计数表 + 资源是否够用」这一模型最直白的题目。

---

## 模式三：双向映射（一一对应）

适用信号：要判断两个序列的对应关系是否「处处一致且互不冲突」。既要检查一个元素映射到的
对方元素是否正确，也要防止两个元素映射到同一个对方元素。核心是用哈希表把映射本身记下来。

### 205. 同构字符串（简单）

**题目**：给定两个字符串 `s` 和 `t`，判断它们是否同构。同构指 `s` 中的每个字符都能被**唯一**替换成 `t` 中对应的字符，且字符的相对顺序不变。例如 `egg` 与 `add` 同构（`e→a`、`g→d`），`foo` 与 `bar` 不同构。

**思路（正反两张哈希表）**：
「同构」要求字符之间是一一对应，有两层约束：

- **正向唯一**：同一个 `s` 字符不能映射到两个不同的 `t` 字符；
- **反向唯一**：两个不同的 `s` 字符不能映射到同一个 `t` 字符。

所以同时维护两张表：`s→t` 和 `t→s`，逐位检查：

1. 若 `s[i]` 已建立映射，必须正好等于 `t[i]`，否则违反正向唯一；
2. 若 `t[i]` 已被别的 `s` 字符占用，则违反反向唯一；
3. 都不冲突，就建立双向映射。

为什么**必须**两张表：只留 `s→t` 会漏掉反向冲突。反例 `s = "ab"`、`t = "aa"`：
`a→a` 合法，`b→a` 在正向表里是新建映射、也没冲突，但 `a` 和 `b` 抢了同一个 `t` 字符 `a`，
必须靠 `t→s` 表发现「`a` 已被 `s` 的 `a` 占用」。

**代码**（`src/hash/isomorphic_strings.py` / `.cpp`）：

```python
def is_isomorphic(s, t):
    if len(s) != len(t):
        return False
    forward = {}
    backward = {}
    for a, b in zip(s, t):
        if forward.get(a, b) != b or backward.get(b, a) != a:
            return False
        forward[a] = b
        backward[b] = a
    return True
```

```cpp
bool isIsomorphic(const std::string &s, const std::string &t) {
    if (s.size() != t.size()) return false;
    std::unordered_map<char, char> forward, backward;
    for (std::size_t i = 0; i < s.size(); ++i) {
        auto fit = forward.find(s[i]);
        if (fit != forward.end() && fit->second != t[i]) return false;
        auto bit = backward.find(t[i]);
        if (bit != backward.end() && bit->second != s[i]) return false;
        forward[s[i]] = t[i];
        backward[t[i]] = s[i];
    }
    return true;
}
```

- **复杂度**：时间 O(n)，空间 O(字符集大小)。
- **易错点**：`forward.get(a, b) != b` 用默认值 `b` 巧妙地同时处理了「没映射」和「映射正确」两种情况；长度不同直接返回 `False`；只写一张表是本题最常见的错误。
- **相似题**：290. 单词规律（把 `s` 的字符对应到 `t` 的单词，套同一套双向映射）；49. 字母异位词分组（从「一一对应」扩展到「多对一的分组指纹」）。

---

## 模式四：指纹分组

适用信号：要把一批元素按「某些与顺序无关的特征是否一致」分成若干组。关键是设计一个
**稳定指纹**：特征相同 ⇒ 指纹相同。

### 49. 字母异位词分组（中等）

**题目**：给定字符串数组 `strs`，把互为字母异位词的字符串分到同一组，返回所有分组。例如 `["eat","tea","tan","ate","nat","bat"]` 分为 `[["eat","tea","ate"], ["tan","nat"], ["bat"]]`。

**思路（排序作指纹 + 哈希分组）**：
互为异位词的两个串，字母多重集合相同。要判断「多重集合是否相同」，最简单的办法是给每个串
算一个与顺序无关的指纹，指纹一样就是一组：

- **排序指纹**：把串里的字符排序，异位词排序后必然完全相同。代码最短；
- **计数指纹**：统计 26 个字母的出现次数，拼成一个长度 26 的 key。省掉排序的 `log`，但写起来啰嗦。

本题用排序指纹。用哈希表 `key -> 该组的字符串列表`，遍历一遍即可完成分组。

为什么排序是合法的指纹：排序会把「同一多重集合的所有排列」映射到同一个有序序列上，
这正是「异位词等价类」的一个完美代表元。

**代码**（`src/hash/group_anagrams.py` / `.cpp`）：

```python
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = "".join(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())
```

```cpp
std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string> strs) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    for (const std::string &s : strs) {
        std::string key = s;
        std::sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    std::vector<std::vector<std::string>> res;
    for (auto &pair : groups) res.push_back(pair.second);
    return res;
}
```

- **复杂度**：设字符串长度 k、个数 n，时间 O(n × k log k)，空间 O(n × k)。若改用计数指纹，时间可降到 O(n × k)。
- **易错点**：返回的顺序任意（组内、组间都不要求），不要依赖哈希表的遍历顺序；空串也能正常作为 key 参与分组。
- **相似题**：242. 有效的字母异位词（两两判断，是本题的「原子操作」）；249. 移位字符串分组（指纹改成「相邻字符差」）；面试里常见的「按特征分组」都可以套这个「算指纹 + 哈希桶」的框架。

---

## 模式五：集合判定（判环与起点扫描）

适用信号：要在无序数据里处理「重复 / 连续 / 相邻」关系，但又不希望排序（排序就是 O(n log n) 了）。
常用套路是把访问过的状态装进集合，靠「在不在」来判断是否重复或是否相邻。

### 202. 快乐数（简单）

**题目**：判断一个正整数 `n` 是否为「快乐数」。定义：把 `n` 替换成它各位数字的平方和，反复进行，如果最终能得到 1，就是快乐数；如果陷入不包含 1 的循环，就不是。例如 `19 → 1²+9²=82 → 68 → 100 → 1`，所以 19 是快乐数。

**思路（哈希集合判环）**：
把「反复求平方和」看作一串状态转移。从任意起点出发，结局只有两种：**到达 1**，
或者**进入一个循环**（永远回不到 1）。判断「会不会重复」正是哈希集合的强项：

- 每一步算出下一个数 `next`；
- 若 `next == 1`，是快乐数；
- 若 `next` 已经出现过，说明进入了循环，不是快乐数；
- 否则把 `next` 记进集合，继续。

为什么需要集合：不记已访问的数，循环会无限进行；集合保证每个状态最多访问一次，
迭代必然终止。数学上这个序列只会落进有限个状态，且要么到 1 要么成环。

**代码**（`src/hash/happy_number.py` / `.cpp`）：

```python
def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1
```

```cpp
int nextNumber(int n) {
    int total = 0;
    while (n > 0) {
        int d = n % 10;
        total += d * d;
        n /= 10;
    }
    return total;
}

bool isHappy(int n) {
    std::unordered_set<int> seen;
    while (n != 1 && seen.find(n) == seen.end()) {
        seen.insert(n);
        n = nextNumber(n);
    }
    return n == 1;
}
```

- **复杂度**：时间 O(log n)（序列长度有上界），空间 O(log n)。
- **易错点**：循环条件同时判「到没到 1」和「是否重复」，缺一不可；先判断再记录，顺序反了会把 1 也塞进集合（本题仍能过，但语义不清晰）。
- **相似题**：141. 环形链表（同一套「判环」思想，链表版用快慢指针可做到 O(1) 空间）；128. 最长连续序列（都用集合记录「见过没」）。

### 128. 最长连续序列（中等）

**题目**：给定未排序的整数数组 `nums`，找出数字连续的最长序列的长度（不要求元素在原数组相邻），要求 O(n)。例如 `[100, 4, 200, 1, 3, 2]` 的最长连续序列是 `[1, 2, 3, 4]`，长度 4。

**思路（哈希集合 + 只从序列起点出发）**：
把所有数装进集合，于是「某个数在不在」是 O(1)。接下来要避免的陷阱是：如果对每个数都往后数一遍
`x+1, x+2, ...`，最坏会数成 O(n²)。

关键观察：**一段连续序列只需要从它的起点数一次**。谁是起点？它的**前一个数 `x-1` 不在集合里**。
于是：

- 若 `x - 1` 在集合里，说明 `x` 是某段序列的中间或末尾，跳过（它会由那段更小的起点统计到）；
- 否则从 `x` 出发，不断查 `x+1, x+2, ...` 直到断掉，这一段长度即为 `1 + 连续延伸的个数`。

为什么整体是 O(n)：每个数要么作为非起点被跳过（O(1)），要么被某段起点向后扫描访问。
段与段互不重叠，一个数不会被两段扫描重复访问，所以扫描的总步数是 O(n)。

**代码**（`src/hash/longest_consecutive_sequence.py` / `.cpp`）：

```python
def longest_consecutive(nums):
    num_set = set(nums)
    best = 0
    for x in num_set:
        if x - 1 in num_set:
            continue
        length = 1
        while x + length in num_set:
            length += 1
        best = max(best, length)
    return best
```

```cpp
int longestConsecutive(const std::vector<int> &nums) {
    std::unordered_set<int> numSet(nums.begin(), nums.end());
    int best = 0;
    for (int x : numSet) {
        if (numSet.count(x - 1)) continue;
        int length = 1;
        while (numSet.count(x + length)) ++length;
        best = std::max(best, length);
    }
    return best;
}
```

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：遍历的是**去重后的集合**而不是原数组过一遍——即使遍历原数组，重复值也只是多几次「非起点」判断，不影响正确性，但集合更干净；「起点判断」`x - 1 not in set` 是保证线性的关键，去掉就退化成 O(n²)；空数组返回 0。
- **相似题**：674. 最长连续递增序列（在**数组顺序**上连续，用线性 DP）；298. 二叉树最长连续序列（树上版，DFS 传前驱值）。三者形式相似，但「连续」的含义不同，要注意区分（数组相邻 / 数值相邻 / 树上父子）。

---

## 规律总结

1. **哈希表的第一性问题**：能不能把「枚举两个元素核对关系」改写成「对每个元素，查它需要的那个值在不在」？能，就能从 O(n²) 降到 O(n)。1、242、49、128 全是这个变换的不同外衣。
2. **先查后存**是「元素不重复使用」的通用保证。1 里若先存后查，`[3, 3]` 会自己配自己；560 里先累加答案再记录当前前缀和，才不会把「自己配自己」的空区间算进去。
3. **区分集合与计数**：只问「有没有 / 会不会重复」就用集合（349、202、128），问「有几个」就用计数字典（242、383、454、560）。选错会导致去重过头或次数丢失。
4. **记录中间结果，把两次枚举接起来**：454 把四个数组劈成两半，一半的和记进表、另一半去查；560 把前缀和记进表、当前前缀和去查目标。凡是「两侧各扫一遍再匹配」的场景，都可以想想哈希表能不能当中间桥。
5. **集合去重的副作用可以利用**：349 边收集边删除，把「结果唯一」顺手做掉；128 用集合去重顺便加快判定；202 用集合把无限迭代变成有限。
6. **一一对应要双向检查**：205 只维护 `s→t` 会漏掉「两个字符抢同一个目标」的情况，必须补上反向表。凡是「映射/替换/规律」题，都要问一句：反向唯一吗？
7. **指纹的精髓是「无关顺序、相同即等价」**。排序、计数、相邻差都能当指纹，选最贴合题目约束的那个。49 用排序指纹最省事，字符集固定时用计数指纹更快。
8. **线性两层循环的证明**：128 里每个元素只被扫描一次，靠的是「只从起点出发」。凡是「看起来有两层循环却声称 O(n)」的哈希题，一定有一句类似「每个元素最多被处理常数次」的话要讲清楚。
9. **哈希表不解决排序问题，也不买单调性**：需要「有序」「下一个更大」这类信息时（如 167 有序两数之和、滑动窗口），哈希表帮不上，要换成双指针或单调栈；560 之所以能用哈希解，正是因为数组带负数、前缀和不单调，滑动窗口反而失效。
