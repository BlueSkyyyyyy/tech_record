# 字符串匹配：KMP、滚动哈希与 Manacher

「一个串里找另一个串」「找最长重复子串」「找最长回文」——这类问题的共同特点是：**朴素
做法要在每个位置重新比较一遍，重复劳动多**。本篇讲的三种工具，都是在「不重复比较已有
信息」这件事上做文章：

- **KMP 前缀函数**：记住模式串自己的「最长相等前后缀」，失配时只回退模式串指针，主串不
  回头。它还能顺手判定周期、求最长快乐前缀、构造最短回文串。
- **滚动哈希（Rabin-Karp）**：把一段子串编码成一个数，窗口右移时 O(1) 更新，于是「两个
  子串是否相同」变成「两个数是否相同」，配合二分答案能求最长重复子串。
- **Manacher**：利用回文的对称性，让已经探明的回文半径被后面的中心复用，把「每个中心
  往两边扩」的 O(n²) 压到 O(n)。

它们和第 15 篇「字符串」是同一片领域，但视角不同：那篇偏「字符串怎么加工」（反转、切分、
进制转换），本篇偏「怎么在一堆子串里做匹配与查找」。

本篇 10 题按工具分五组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：前缀函数（KMP） | 28. 找出字符串中第一个匹配项 / 1392. 最长快乐前缀 / 214. 最短回文串 | 简单 / 中等 / 困难 |
| 模式二：用前缀函数判周期 | 459. 重复的子字符串 | 简单 |
| 模式三：滚动哈希（Rabin-Karp） | 187. 重复的 DNA 序列 / 1044. 最长重复子串 | 中等 / 困难 |
| 模式四：Manacher | 5. 最长回文子串 / 647. 回文子串 | 中等 / 中等 |
| 模式五：字符串包含与拼接 | 796. 旋转字符串 / 686. 重复叠加字符串匹配 | 简单 / 中等 |

> 一句话记住匹配三件套：**KMP 用「模式串自己的前后缀」省回退；滚动哈希用「数字指纹」
> 省比较；Manacher 用「对称复用」省扫描。**

---

## 模式一：前缀函数（KMP）

**适用信号**：在主串里找模式串；「最长相等前后缀」本身就是答案的一部分（快乐前缀、最短
回文串、周期判定）。

**核心动作**：先只针对模式串（或原串）构造前缀函数 `lps`。`lps[i]` = 「前 i+1 个字符」
这一段里，最长的「既是前缀又是后缀」的真子串长度。构造它本身就是一个自我匹配的过程。
匹配时主串指针 `i` 只前进不后退，失配就让模式串指针 `j = lps[j-1]`。

### 28. 找出字符串中第一个匹配项的下标（简单）

**题目**：给定 haystack 和 needle，返回 needle 在 haystack 中第一次出现的下标，不存在
返回 -1；needle 为空返回 0。

**思路**：

```python
def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0  # 当前最长相等前后缀的长度
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]  # 前后缀接不上，退而求其次
        else:
            lps[i] = 0
            i += 1
    return lps


def str_str(haystack, needle):
    if needle == "":
        return 0
    lps = build_lps(needle)
    m = len(needle)
    j = 0  # needle 上已经匹配的长度
    for i in range(len(haystack)):
        while j > 0 and haystack[i] != needle[j]:
            j = lps[j - 1]  # 主串指针 i 不回退
        if haystack[i] == needle[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1
```

```cpp
std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

int strStr(const std::string &haystack, const std::string &needle) {
    if (needle.empty()) {
        return 0;
    }
    std::vector<int> lps = buildLps(needle);
    int m = static_cast<int>(needle.size()), j = 0;
    for (int i = 0; i < static_cast<int>(haystack.size()); ++i) {
        while (j > 0 && haystack[i] != needle[j]) {
            j = lps[j - 1];
        }
        if (haystack[i] == needle[j]) {
            ++j;
        }
        if (j == m) {
            return i - m + 1;
        }
    }
    return -1;
}
```

**为什么失配后可以只退模式串指针**：假设模式串前 `j` 个字符已经和主串当前位置之前的
`j` 个字符对齐且相等。若下一位失配，朴素做法是把模式串整体右移一格、从头的第一个字符
重新比。但这段已匹配的内容里，「前 `lps[j-1]` 个字符」和「它的最后 `lps[j-1]` 个字符」
是一样的。也就是说：把模式串往后挪到「当前已匹配段的最后一个相等前缀」处，前面的部分
已经自动对上了，主串指针根本不用动。`lps` 记录的就是这个「挪多少」。

**为什么构造 `lps` 也是同一个回退逻辑**：求 `lps[i]` 时，我们希望知道「以 i 结尾的后缀
能和前缀对上多长」。如果 `pattern[i]` 正好接在已知的 `length` 后面，长度就 +1；否则
`length` 也要回退到 `lps[length-1]` 再试——因为更短的前缀才有可能接得上。整个构造过程
就是「模式串和自己做一次 KMP」。

- **复杂度**：时间 O(n + m)，空间 O(m)。主串指针 `i` 全程只增不减，`j` 虽然会回退，但
  每次回退都对应之前 `j` 的增长，摊还下来总回退次数是 O(n)。
- **易错点**：构造 `lps` 时 `i` 从 1 开始（`lps[0]` 永远是 0）；失配回退用 `while` 而不是
  `if`，因为可能要连续回退；返回下标是 `i - m + 1`，不是 `i`。
- **相似题**：`1392` 直接返回 `lps[-1]` 那段前缀；`214` 把原串和反转串拼起来复用 `lps`；
  `459` 用 `lps` 判周期。第 15 篇「字符串」里的 28 题与本篇同源，可对照看。

### 1392. 最长快乐前缀（中等）

**题目**：「快乐前缀」是既是原串的非空真前缀、又是后缀的字符串。返回最长的那个，没有
则返回空串。

**思路**：

```python
def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def longest_prefix(s):
    if not s:
        return ""
    return s[: build_lps(s)[-1]]
```

```cpp
std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

std::string longestPrefix(const std::string &s) {
    if (s.empty()) {
        return "";
    }
    int k = buildLps(s).back();
    return s.substr(0, k);
}
```

**为什么一步到位**：`lps[n-1]` 的定义就是「整串最长的相等前后缀长度」，而题目要的正是
这个。因为 `lps[i] < i+1`，取到的这一段天然是**真**前缀，不必额外排除整串。这题是「把
KMP 的中间产物当答案」的最典型例子。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：空串要单独返回 `""`；`lps[-1]` 可能是 0，此时切片得到空串，正好正确。
- **相似题**：`214` 求的是「最长回文前缀」，思路是把它转化成前后缀问题后用同一个 `lps`；
  `459` 也只用到 `lps` 的最后一个值。

### 214. 最短回文串（困难）

**题目**：可以在 s 前面添加字符使其变成回文串，返回构造出的最短回文串。

**思路**：

```python
def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def shortest_palindrome(s):
    if len(s) <= 1:
        return s
    combined = s + "#" + s[::-1]
    k = build_lps(combined)[-1]
    return s[k:][::-1] + s
```

```cpp
std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

std::string shortestPalindrome(const std::string &s) {
    if (s.size() <= 1) {
        return s;
    }
    std::string rev(s.rbegin(), s.rend());
    int k = buildLps(s + "#" + rev).back();
    std::string tail = s.substr(k);
    std::reverse(tail.begin(), tail.end());
    return tail + s;
}
```

**为什么这样拼**：答案一定是「补一段 + s」。要让总长度最短，就要让 s 有一段尽可能长的
**回文前缀**——这段可以原样留在中间，只需把剩下的后缀反过来补在前面。于是问题变成
「s 的最长回文前缀有多长」。

把 `combined = s + '#' + reverse(s)` 求 `lps[-1] = k`。理由：`combined` 的一个后缀必须
落在 `reverse(s)` 部分，它是「s 的某个前缀的逆序」；而它又作为前缀出现在 s 开头。两者
相等，等价于 s 的开头这段正着读、反着读一样，即回文。分隔符 `'#'` 保证匹配不越过边界。
得到 `k` 后，答案就是 `reverse(s[k:]) + s`。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：拼接时必须加分隔符，否则前后缀可能跨过 s 与 reverse(s) 的边界，`k` 会偏大；
  `s[k:]` 要逆序后再拼到前面，别忘了反转。
- **相似题**：`1392` 是本题的「同一工具、不同问法」；回文类的 `5`/`647` 是另一条线
  （中心扩展 / Manacher），见模式四。

---

## 模式二：用前缀函数判周期

**适用信号**：一个串能不能由某个小段重复若干次拼成。

**核心动作**：先求 `lps[-1] = L`（最长相等前后缀），最小周期 `p = n - L`。若 `L > 0` 且
`n % p == 0`，就是重复串。

### 459. 重复的子字符串（简单）

**题目**：判断非空字符串 s 能否由它的某个子串重复多次构成。

**思路**：

```python
def build_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length > 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    return lps


def repeated_substring_pattern(s):
    n = len(s)
    lps = build_lps(s)
    period = n - lps[-1]
    return lps[-1] > 0 and n % period == 0
```

```cpp
std::vector<int> buildLps(const std::string &pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int length = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[length]) {
            lps[i] = ++length;
            ++i;
        } else if (length > 0) {
            length = lps[length - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }
    return lps;
}

bool repeatedSubstringPattern(const std::string &s) {
    int n = static_cast<int>(s.size());
    int last = buildLps(s).back();
    int period = n - last;
    return last > 0 && n % period == 0;
}
```

**为什么 `n - lps[-1]` 是最小周期**：如果 s 由长度 `p` 的块重复而成，那么把 s 向右挪 `p`
位，后半段和前半段逐字重合，重合长度是 `n - p`，所以最长相等前后缀 `L ≥ n - p`。反过来，
取最长的 `L`，`p = n - L` 就是「能对齐的最短位移」，也就是最小周期。**必须再检查
`n % p == 0`**：例如 `"abababa"` 的 `L = 5`、`p = 2`，但长度 7 不是 2 的整数倍，它不是
块重复。`L > 0` 则排除 `"abcd"` 这种 `L = 0, p = n` 被 `n % n == 0` 误判的情况。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：`L > 0` 和 `n % p == 0` 两个条件缺一不可；本题只判「能否」，若要返回最小
  周期块，就直接取 `s[:p]`。
- **相似题**：`28` 是 `lps` 的主场；`1392`、`214` 都只用 `lps[-1]`，和本题同一用法。
  第 15 篇的 459 与本篇一致，可对照。

---

## 模式三：滚动哈希（Rabin-Karp）

**适用信号**：要反复比较「很多个等长子串是否相同」，或者「两个串的最长公共子串」。

**核心动作**：把子串按多项式编码成一个数（令牌），窗口滑动时 O(1) 更新；用哈希值去重，
出现重复时再对真实子串比对一次排除碰撞。

### 187. 重复的 DNA 序列（中等）

**题目**：找出所有长度为 10、出现次数超过一次的子串。

**思路**：

```python
def find_repeated_dna_sequences(s):
    length = 10
    if len(s) < length + 1:
        return []
    seen = set()
    added = set()
    res = []
    for i in range(len(s) - length + 1):
        sub = s[i : i + length]
        if sub in seen and sub not in added:
            res.append(sub)
            added.add(sub)
        seen.add(sub)
    return res
```

```cpp
std::vector<std::string> findRepeatedDnaSequences(const std::string &s) {
    const int length = 10;
    std::vector<std::string> res;
    if (static_cast<int>(s.size()) < length + 1) {
        return res;
    }
    std::unordered_set<std::string> seen, added;
    for (int i = 0; i + length <= static_cast<int>(s.size()); ++i) {
        std::string sub = s.substr(i, length);
        if (seen.count(sub) && !added.count(sub)) {
            res.push_back(sub);
            added.insert(sub);
        }
        seen.insert(sub);
    }
    return res;
}
```

**为什么用两个集合**：`seen` 回答「这个子串之前出现过吗」，`added` 回答「这个子串已经
收进答案了吗」。只用 `seen` 的话，同一个子串出现三次会被加入两次，所以需要 `added` 去重。
窗口长度固定为 10，直接切片最直观；若窗口很长、字符集很大，切片本身要 O(L) 复制，这时
用滚动哈希把每个窗口压成一个数、只存数字，就能把每步比较降到 O(1)。

- **复杂度**：时间 O(n·L)（切片复制），空间 O(n·L)；改用滚动哈希可到时间 O(n)、空间
  O(n)。
- **易错点**：循环上界是 `len(s) - length + 1`，别漏掉最后一个窗口；`added` 去重别忘。
- **相似题**：`1044` 把同一套「窗口 + 令牌」升级成「二分长度 + 滚动哈希」；第 15 篇的
  字符串匹配 / 第 17 篇前缀树都能做类似的多模式查找，但思路不同。

### 1044. 最长重复子串（困难）

**题目**：找出出现次数不少于两次的最长子串（允许重叠），返回任意一个。

**思路**：

```python
def longest_dup_substring(s):
    n = len(s)
    if n < 2:
        return ""
    base, mod = 131, (1 << 61) - 1

    def check(length):
        # 返回一个长度为 length 的重复子串；不存在返回 None
        if length == 0:
            return ""
        power = pow(base, length - 1, mod)
        h = 0
        for i in range(length):
            h = (h * base + ord(s[i])) % mod
        seen = {h: 0}
        for i in range(1, n - length + 1):
            h = ((h - ord(s[i - 1]) * power) % mod * base + ord(s[i + length - 1])) % mod
            if h in seen and s[seen[h] : seen[h] + length] == s[i : i + length]:
                return s[i : i + length]
            seen[h] = i
        return None

    lo, hi = 0, n - 1
    ans = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        found = check(mid)
        if found is not None:
            ans = found
            lo = mid + 1
        else:
            hi = mid - 1
    return ans
```

```cpp
const long long MOD = 1000000007LL;
const long long BASE = 131LL;

bool check(int length, const std::string &s, const std::vector<long long> &pw,
           std::string &out) {
    if (length == 0) {
        out = "";
        return true;
    }
    int n = static_cast<int>(s.size());
    long long power = pw[length - 1];
    long long h = 0;
    for (int i = 0; i < length; ++i) {
        h = (h * BASE + static_cast<unsigned char>(s[i])) % MOD;
    }
    std::unordered_map<long long, int> seen;
    seen[h] = 0;
    for (int i = 1; i + length <= n; ++i) {
        h = ((h - static_cast<unsigned char>(s[i - 1]) * power) % MOD + MOD) % MOD;
        h = (h * BASE + static_cast<unsigned char>(s[i + length - 1])) % MOD;
        auto it = seen.find(h);
        if (it != seen.end() &&
            s.compare(it->second, length, s, i, length) == 0) {
            out = s.substr(i, length);
            return true;
        }
        seen[h] = i;
    }
    return false;
}

std::string longestDupSubstring(const std::string &s) {
    int n = static_cast<int>(s.size());
    if (n < 2) {
        return "";
    }
    std::vector<long long> pw(n, 1);
    for (int i = 1; i < n; ++i) {
        pw[i] = pw[i - 1] * BASE % MOD;
    }
    int lo = 0, hi = n - 1;
    std::string ans;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        std::string out;
        if (check(mid, s, pw, out)) {
            ans = out;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return ans;
}
```

**为什么可以二分长度**：如果存在长度为 L 的重复子串，那么取它的前缀，就得到长度 L-1
的重复子串。所以「能否重复」关于长度是单调的（L 可行则更短的都可行），可以二分找出最大
的 L。判定 `check(L)` 就是「把所有长度为 L 的窗口的哈希值放进集合，看有没有重复」。

**为什么哈希能滑动更新**：长度为 L 的窗口哈希是
`s[i]·base^(L-1) + s[i+1]·base^(L-2) + … + s[i+L-1]`。窗口右移一位时，最高位
`s[i]·base^(L-1)` 被减去，整体乘 `base`，再补上最低位的新字符，就得到新窗口的哈希，
不必重算整段。代码里 `power = base^(L-1)` 就是用来减去最高位的。

**为什么哈希重复后还要比对真实子串**：不同子串可能算出同一个哈希（碰撞）。比对一次真实
字符，既能确认答案正确，也让代码不依赖「哈希绝对无碰撞」这一不太靠谱的假设。Python 版
用 `2^61-1` 这个大模数，C++ 版用 `10^9+7`，碰撞概率都很低。

- **复杂度**：时间 O(n log n)，空间 O(n)。
- **易错点**：Java/C++ 里滚动哈希的减法要处理负数（`+MOD` 再取模，或先加）；二分要保留
  「找到的具体串」而不是只留布尔值，否则最后无法返回正确答案（本题在 `check` 里直接返回
  找到的串）；`check(0)` 视为可行（空串），作为二分下界。
- **相似题**：`187` 是定长窗口版本；`718` 最长重复子数组可用滚动哈希 + 二分，也可用 DP；
  `1044` 与第 15 篇的「前缀函数」互为一体两面。

---

## 模式四：Manacher

**适用信号**：最长回文子串、回文子串计数。

**核心动作**：把原串插成 `^#a#b#…#$`，让奇偶回文统一；维护向右延伸最远的回文
`[center, right]`，用对称点 `2*center - i` 的回文半径给当前中心一个起点，再从该起点继续
向两边扩。

### 5. 最长回文子串（中等）

**题目**：返回 s 中最长的回文子串。

**思路**：

```python
def longest_palindrome(s):
    if len(s) <= 1:
        return s
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(1, n - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2
    return s[start : start + max_len]
```

```cpp
std::string longestPalindrome(const std::string &s) {
    if (s.size() <= 1) {
        return s;
    }
    std::string t = "^#";
    for (char c : s) {
        t += c;
        t += '#';
    }
    t += "$";
    int n = static_cast<int>(t.size());
    std::vector<int> p(n, 0);
    int center = 0, right = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (i < right) {
            p[i] = std::min(right - i, p[2 * center - i]);
        }
        while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
            ++p[i];
        }
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }
    int maxLen = 0, centerIdx = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] > maxLen) {
            maxLen = p[i];
            centerIdx = i;
        }
    }
    int start = (centerIdx - maxLen) / 2;
    return s.substr(start, maxLen);
}
```

**为什么能线性**：维护一个「向右伸得最远」的回文，其右端点为 `right`。当处理到 `i < right`
时，`i` 落在该回文内部，它关于 `center` 的对称点 `j = 2*center - i` 已经算过。若 `j` 的
回文完全落在 `[.., right]` 内，那么 `i` 的回文边界与 `j` 完全对称，可直接照搬；若超出边界，
则只能确定到 `right` 为止，从 `right` 继续往外扩。于是除「扩边界」外几乎没有重复比较，
总复杂度 O(n)。

**为什么 `p[i]` 恰好等于原串回文长度**：插入 `#` 后每个回文都是奇数长度，半径每增加 1
对应原串各加一个字符，所以 `p[i]` 的数值就是原串里该回文的长度。反推起点：
`start = (center_idx - max_len) // 2`。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：首尾哨兵 `^`、`$` 必不可少，否则 `while` 要额外判边界；插串后 `i` 从 1 到
  `n-2`；反推起点别忘记除以 2。
- **相似题**：`647` 用同一套半径直接求和；`214` 求最短回文串也能用 Manacher，但用 KMP
  更简洁；第 13 篇「动态规划」的 5/647 是从「区间 DP」的角度讲的，可对照两种视角。

### 647. 回文子串（中等）

**题目**：统计字符串里回文子串的个数（按位置计数）。

**思路**：

```python
def count_substrings(s):
    if not s:
        return 0
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(1, n - 1):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    total = 0
    for i in range(1, n - 1):
        if i % 2 == 0:
            total += p[i] // 2 + 1  # 原串字符为中心：奇数长度
        else:
            total += (p[i] + 1) // 2  # '#' 为中心：偶数长度
    return total
```

```cpp
int countSubstrings(const std::string &s) {
    if (s.empty()) {
        return 0;
    }
    std::string t = "^#";
    for (char c : s) {
        t += c;
        t += '#';
    }
    t += "$";
    int n = static_cast<int>(t.size());
    std::vector<int> p(n, 0);
    int center = 0, right = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (i < right) {
            p[i] = std::min(right - i, p[2 * center - i]);
        }
        while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
            ++p[i];
        }
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }
    int total = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (i % 2 == 0) {
            total += p[i] / 2 + 1;  // 原串字符为中心：奇数长度
        } else {
            total += (p[i] + 1) / 2;  // '#' 为中心：偶数长度
        }
    }
    return total;
}
```

**为什么按中心分类计数**：回文都由「中心」唯一确定。在插了 `#` 的 t 里，中心分两类：

- **偶数下标**是原串字符，对应**奇数长度**回文，半径可以是 0、2、4…（每两级对应长度
  1、3、5…），共 `p[i]//2 + 1` 个；
- **奇数下标**是 `#`，对应**偶数长度**回文，半径可以是 1、3、5…（长度 2、4、6…），共
  `(p[i]+1)//2` 个。

把所有中心的贡献相加，就是回文子串总数。这样避免了「对每个中心重新往外扩」的 O(n²)。

- **复杂度**：时间 O(n)，空间 O(n)。
- **易错点**：空串要单独返回 0，否则退化出的 `"^##$"` 会多算一个 `#` 中心；奇偶中心
  的计数公式不同，别写反。
- **相似题**：`5` 只是把「求和」换成「求最大半径」；`516` 最长回文子序列用的是区间 DP，
  和「子串」不同（子序列可不连续）。

---

## 模式五：字符串包含与拼接

**适用信号**：判断「一个串能否通过旋转 / 重复拼接得到另一个串」。

**核心动作**：把「所有可能的拼接结果」显式构造出来（`s + s` 或重复若干次），再用子串
查找一次判断。

### 796. 旋转字符串（简单）

**题目**：判断 s 能否经过若干次「把最左边字符移到最右边」变成 goal。

**思路**：

```python
def rotate_string(s, goal):
    return len(s) == len(goal) and goal in s + s
```

```cpp
bool rotateString(const std::string &s, const std::string &goal) {
    return s.size() == goal.size() && (s + s).find(goal) != std::string::npos;
}
```

**为什么 `s + s` 能覆盖所有旋转**：s 的每一个旋转，都是 `s + s` 里某个长度为 n 的窗口。
例如 `s = "abcde"`，`s + s = "abcdeabcde"`，从下标 0 到 4 各取长度 5，正好得到五种旋转。
所以只需先比长度，再看 goal 是否在 `s + s` 中。长度必须先比：否则 `s = "abc"`、
`goal = "abcabc"` 时 goal 会落在 `s+s` 里，但长度不同并非旋转。

- **复杂度**：时间 O(n)（标准子串查找），空间 O(n)。
- **易错点**：先判长度相等；空串对空串视为可旋转。
- **相似题**：`686` 是「重复叠加」版本；`28` 的子串查找是这里的底层工具。

### 686. 重复叠加字符串匹配（中等）

**题目**：返回 a 需要重复叠加的最小次数，使 b 成为叠加结果的子串；做不到返回 -1。

**思路**：

```python
def repeated_string_match(a, b):
    import math

    times = math.ceil(len(b) / len(a))
    for t in (times, times + 1):
        if b in a * t:
            return t
    return -1
```

```cpp
int repeatedStringMatch(const std::string &a, const std::string &b) {
    int times = static_cast<int>((b.size() + a.size() - 1) / a.size());
    for (int t = times; t <= times + 1; ++t) {
        std::string rep;
        for (int i = 0; i < t; ++i) {
            rep += a;
        }
        if (rep.find(b) != std::string::npos) {
            return t;
        }
    }
    return -1;
}
```

**为什么只需要试 `t` 和 `t+1`**：设 `t = ceil(len(b) / len(a))` 是让叠加串长度不小于 b
所需的最少次数。若答案存在，它只可能是 `t`（起点刚好落在某次重复的开头附近）或 `t + 1`
（起点落在某次重复中间、长度刚好差一点）。次数再多也不会让 b 首次出现：因为 b 长度固定，
任何一次成功匹配所用的窗口都落在连续 `t+1` 个 a 之内，超出部分没有新信息。所以试两次
即可。

- **复杂度**：时间 O((n + m)·m) 量级（每轮子串查找），空间 O(n + m)；用 KMP 查找可线性。
- **易错点**：`times` 要用向上取整（`(len(b) + len(a) - 1) // len(a)`）；必须试 `t+1`，
  只试 `t` 会漏解（如 `a="abcd", b="cdabcdab"`）；都不含则返回 -1。
- **相似题**：`796` 是次数固定为「一次旋转」的特例；`28` 的 KMP 可替换这里的 `find` 以
  获得线性复杂度。

---

## 规律总结

1. **匹配问题的核心矛盾是「别重复比较已有的信息」**。KMP 利用模式串自己的前后缀、滚动
   哈希利用窗口之间的重叠、Manacher 利用回文的对称性，本质都是把已经算过的结果留下来
   给后面用。遇到卡在 O(n·m) 的匹配题，先问：**哪部分信息其实可以复用？**

2. **前缀函数 `lps` 是「一鱼多吃」的中间产物**。`lps[i]` = 最长相等前后缀长度；它是
   KMP 失配时的回退表，直接就能回答 `1392` 最长快乐前缀、`459` 的周期问题，配上拼接
   技巧还能解 `214` 最短回文串。凡是问题里出现「前缀 / 后缀相等」字样，先求 `lps`。

3. **构造 `lps` 与使用 `lps` 是同一套回退逻辑**。都是用 `while` 连续回退到
   `lps[length-1]` 再尝试匹配。写的时候只要记住：能接上就 `+1`，接不上就回退，回退到
   0 还不行就置 0。

4. **判周期公式：`p = n - lps[-1]`，且要 `lps[-1] > 0 && n % p == 0`**。两个条件缺一
   不可：`n % p == 0` 挡住「有公共前后缀但长度对不齐」的情况，`lps[-1] > 0` 挡住「没有
   公共前后缀、`p = n`」被 `n % n == 0` 误判。本题思想上和「最小循环节」完全一致。

5. **滚动哈希把「比较两段子串」变成「比较两个数」**。窗口右移的更新式是
   「减去最高位 `s[i]·base^(L-1)`，整体乘 `base`，加上新字符」。哈希碰撞不能无视，命中
   哈希后**再比对一次真实子串**，既保证正确又不牺牲平均速度。

6. **长度单调的问题可以「二分答案 + 哈希判定」**。`1044` 就是这样：存在长度 L 的重复
   子串 ⟹ 存在长度 L-1 的（取前缀），于是二分答案，每轮用滚动哈希集合判重。这句「取前缀
   仍成立」是能二分的理由，遇到「最长 / 最小且可单调验证」的题都可以套。

7. **Manacher 的三件套：插 `#`、写半径、用对称点**。插 `#` 统一奇偶；`p[i]` 记半径；
   `p[i] = min(right - i, p[2*center - i])` 是加速的关键——在旧回文内就先「照搬对称
   点的答案，但不超过右边界」，出了边界再老实扩。`5` 取最大半径、`647` 按中心累加即可。

8. **已知回文能贡献多少个子串要按中心奇偶分开算**。原串字符为中心的是奇数长度回文，
   贡献 `p[i]//2 + 1`；`#` 为中心的是偶数长度回文，贡献 `(p[i]+1)//2`。空串是退化情况，
   单独 `return 0`。

9. **「能否通过旋转 / 拼接得到」常常可以显式构造再查找**。`796` 用 `s + s` 一次覆盖所有
   旋转；`686` 只需试 `ceil(len(b)/len(a))` 和它加一两种次数。先想清楚「所有候选答案
   长什么样」，再拿现成的子串查找一次判定，往往比手写模拟短得多。

10. **和第 15 篇「字符串」、第 13 篇「动态规划」有交叉**。`28`、`459` 在第 15 篇也出现，
    那里偏「反转 / 切分 / 解析」，这里偏「匹配算法」；`5`、`647`、`516` 既可用 Manacher /
    中心扩展，也可用区间 DP，两条路线各有取舍（线性 vs 好理解）。把同一题在不同篇里
    对照着看，能更清楚「算法选择」而不是「唯一答案」。

11. **先估长度再选工具**。字符集小、窗口短，直接切片 + 哈希集合就够（`187`）；窗口长、
    要反复比较，用滚动哈希（`1044`）；只关心前后缀关系，用 `lps`；只关心回文，用
    Manacher。工具本身不复杂，难的是从题面的「要找什么」对上「该用哪一把」。
