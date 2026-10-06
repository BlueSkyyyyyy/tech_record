# 状态压缩：把集合压进一个整数

有一类问题的状态本身就是"一个集合"：哪些元素被选了、哪些技能被覆盖了、哪些点被
访问过了。如果老老实实用一个 `set` 或布尔数组去存状态，既难当数组下标，也难做记忆化，
状态之间还没法快速比较。**状态压缩**就是把这些集合塞进一个整数的二进制位里——第 i 位
为 1 表示"元素 i 在集合中"。一个整数就代表一个集合。

压进去以后，集合运算全变成了位运算，快且干净：

| 集合操作 | 位运算 |
|---|---|
| 判断元素 `i` 在不在集合 | `mask >> i & 1` |
| 加入元素 `i` | `mask \| (1 << i)` |
| 删除元素 `i` | `mask & ~(1 << i)` 或 `mask ^ (1 << i)`（确定在的话） |
| 两个集合求并 | `m1 \| m2` |
| 两个集合求交 | `m1 & m2` |
| 两个集合是否不冲突 | `m1 & m2 == 0` |
| 集合是否装满 | `mask == (1 << n) - 1` |

能用状态压缩的前提很直接：**元素个数少（一般 ≤ 20），且每个元素只有"在/不在"两种
状态**。因为状态总数是 2^n，n 一大就爆炸。所以看到"n ≤ 12 / 16 / 20"这种小上界，
再配合"选一个子集""覆盖一个集合""访问所有点"这类字眼，就该想到它。

本篇 10 题按"状态用来干什么"分六组：

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：位掩码当开关表（枚举子集） | 78. 子集 / 784. 字母大小写全排列 | 中等 / 中等 |
| 模式二：掩码做冲突校验 + 选或不选 | 1239. 串联字符串的最大长度 / 1255. 得分最高的单词集合 | 中等 / 困难 |
| 模式三：状态 = 已覆盖集合 | 691. 贴纸拼词 / 1125. 最小的必要团队 | 困难 / 困难 |
| 模式四：枚举请求子集做校验 | 1601. 最多可达成的换楼请求数目 | 困难 |
| 模式五：状态 = (当前点, 已访问集合) | 847. 访问所有节点的最短路径 / 943. 最短超级串 | 困难 / 困难 |
| 模式六：状态 = 已分配对象的集合 | 1434. 每个人戴不同帽子的方案数 | 困难 |

> 一句话记住状态压缩：**把"哪些元素被选/被覆盖/被访问"写成一个整数的二进制位，
> 于是"集合的状态"变成"数组的下标"，集合运算变成位运算。**

---

## 模式一：位掩码当开关表（枚举子集）

**适用信号**：要枚举一个 n 元素集合的所有子集，n 不大。

**核心动作**：一个子集就是一个 `0 .. 2^n - 1` 的整数，第 i 位决定第 i 个元素选不选。
`for mask in range(1 << n)` 就把所有子集走了一遍。真正需要"做决定"的可能只有一部分
元素，那就先把这些元素的下标挑出来，只对它们开开关。

### 78. 子集（中等）

**题目**：给定不含重复元素的整数数组 nums，返回它的所有子集（幂集）。

**思路**：

```python
def subsets(nums):
    n = len(nums)
    res = []
    for mask in range(1 << n):
        res.append([nums[i] for i in range(n) if mask >> i & 1])
    return res
```

```cpp
std::vector<std::vector<int>> subsets(const std::vector<int> &nums) {
    int n = static_cast<int>(nums.size());
    std::vector<std::vector<int>> res;
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> subset;
        for (int i = 0; i < n; ++i) {
            if (mask >> i & 1) {
                subset.push_back(nums[i]);
            }
        }
        res.push_back(subset);
    }
    return res;
}
```

**为什么是 2^n 个**：每个元素独立地"选或不选"，n 个元素就是 n 个二元决定，连乘得
2^n。而每个 `0 .. 2^n - 1` 的整数，二进制恰好是一串长度为 n 的二进制决定，二者一一
对应。所以枚举整数 = 枚举子集，不需要递归。

这是回溯篇 78 题的"另一种视角"：回溯是"逐个元素做决定、走一遍决策树"；位枚举是
"直接列出所有决定的结果"。同一组答案，位枚举少了递归、写起来更短，代价是只能处理
n 不太大的情况（2^n 必须能枚举完）。

- **复杂度**：时间 O(n · 2^n)（2^n 个子集，每个花 O(n) 组装），空间 O(n · 2^n)（结果本身）。
- **易错点**：`1 << n` 是子集总数，别写成 `1 << (n - 1)`；取第 i 位用 `mask >> i & 1`，
  位运算优先级低于比较，写 `(mask >> i) & 1` 更保险。
- **相似题**：784（下题）只对字母开开关；1239 / 1255 都是在子集枚举上再加一个"合不合法"
  的校验；位运算篇的 78「位枚举法」与本篇完全同源，可以对照着看。

### 784. 字母大小写全排列（中等）

**题目**：给定字符串 s，其中的字母可以任意改成大写或小写，数字保持不变，返回所有能
得到的字符串。

**思路**：

```python
def letter_case_permutation(s):
    letters = [i for i, ch in enumerate(s) if ch.isalpha()]
    res = []
    for mask in range(1 << len(letters)):
        chars = list(s)
        for j, i in enumerate(letters):
            chars[i] = chars[i].upper() if mask >> j & 1 else chars[i].lower()
        res.append("".join(chars))
    return res
```

```cpp
std::vector<std::string> letterCasePermutation(std::string s) {
    std::vector<int> letters;
    for (int i = 0; i < static_cast<int>(s.size()); ++i) {
        if (std::isalpha(static_cast<unsigned char>(s[i]))) {
            letters.push_back(i);
        }
    }
    int L = static_cast<int>(letters.size());
    std::vector<std::string> res;
    for (int mask = 0; mask < (1 << L); ++mask) {
        std::string cur = s;
        for (int j = 0; j < L; ++j) {
            if (mask >> j & 1) {
                cur[letters[j]] = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(cur[letters[j]])));
            } else {
                cur[letters[j]] = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(cur[letters[j]])));
            }
        }
        res.push_back(cur);
    }
    return res;
}
```

**为什么先收集 letters**：真正有"两种选择"的只有字母，数字的位置永远不动。把字母的
下标单独收集成 `letters`，mask 的第 j 位就表示"第 j 个字母要不要变大写"。这样 mask
的有效位数等于字母个数 L，答案数就是 2^L，而不是把数字也算进去白白浪费。这个"先把
可选对象挑出来"的动作，是所有位枚举模板的起手式。

- **复杂度**：时间 O(2^L · n)（L 是字母个数，每个结果要复制并改写整串），空间 O(2^L · n)。
- **易错点**：别忘了先把字母统一成一种基准（这里用 `.lower()`），否则原串里本来是大写的
  字母会重复；`isalpha` 判断字母，别漏了大小写之外的字符。
- **相似题**：78（上题）是最纯粹的"每个元素选不选"；本题的"每个字母两种写法"是同一
  开关表的直接套用。字符串篇的 6 Z 字形变换也有"行列坐标映射"，但那是模拟而非枚举。

---

## 模式二：掩码做冲突校验 + 选或不选

**适用信号**：从一堆物品里选一个子集，物品之间有"能不能共存"的约束。

**核心动作**：给每个物品算一个掩码（它占用了哪些字母/资源），两个物品能共存当且仅当
掩码不相交（`m1 & m2 == 0`）。于是问题退化成"在不冲突的物品里选一组，使某个目标最优"，
用 0/1 背包式的状态扩展来做。

### 1239. 串联字符串的最大长度（中等）

**题目**：给定字符串数组 arr，从中选出一个子序列拼起来，要求拼出的字符串里没有重复
字符，求能拼出的最大长度。

**思路**：

```python
def max_length(arr):
    # 预先把每个字符串压成 (掩码, 长度)；自带重复字符的直接丢弃。
    items = []
    for s in arr:
        mask = 0
        ok = True
        for ch in s:
            bit = 1 << (ord(ch) - 97)
            if mask & bit:
                ok = False
                break
            mask |= bit
        if ok:
            items.append((mask, len(s)))

    dp = {0: 0}
    for mask, length in items:
        for cur, total in list(dp.items()):
            if cur & mask == 0:
                merged = cur | mask
                cand = total + length
                if dp.get(merged, -1) < cand:
                    dp[merged] = cand
    return max(dp.values())
```

```cpp
int maxLength(const std::vector<std::string> &arr) {
    std::vector<std::pair<int, int>> items;
    for (const std::string &s : arr) {
        int mask = 0;
        bool ok = true;
        for (char ch : s) {
            int bit = 1 << (ch - 'a');
            if (mask & bit) {
                ok = false;
                break;
            }
            mask |= bit;
        }
        if (ok) {
            items.push_back({mask, static_cast<int>(s.size())});
        }
    }

    std::unordered_map<int, int> dp;
    dp[0] = 0;
    for (auto &[mask, length] : items) {
        std::vector<std::pair<int, int>> snapshot(dp.begin(), dp.end());
        for (auto &[cur, total] : snapshot) {
            if ((cur & mask) == 0) {
                int merged = cur | mask;
                int cand = total + length;
                auto it = dp.find(merged);
                if (it == dp.end() || it->second < cand) {
                    dp[merged] = cand;
                }
            }
        }
    }
    int best = 0;
    for (auto &[k, v] : dp) {
        best = std::max(best, v);
    }
    return best;
}
```

**为什么用掩码判冲突**：拼起来不能有重复字符，等价于"选中的每个字符串内部无重复，
且两两使用的字母不重叠"。内部无重复在预处理时就能查掉（自带重复词的直接弃用），
两两不重叠用 `cur & mask == 0` 一次位与判断。`dp` 的键是"已经用掉的字母集合"，
值是"该集合下能达到的最大长度"。

**为什么先存快照**：C++ 里在遍历 `dp` 的同时又往里插入新元素，会让迭代器失效；先把
`dp` 拷一份快照，再基于快照更新。Python 里同理，遍历 `list(dp.items())` 而不要直接
遍历 `dp.items()`。这也是所有"在字典上做 0/1 扩展"的通用注意点。

- **复杂度**：时间 O(n · 2^n)（最多 n 个不同的可达集合），空间 O(2^n)。
- **易错点**：自带重复字符的字符串必须丢弃，不能只靠"和别人的掩码不相交"来保证；
  `dp` 用"集合 → 最大值"而不是布尔可达，否则统计不了长度。
- **相似题**：1255（下题）把"字母不能重复"换成"字母不能超量"；1255 与 691 都是
  "子集/覆盖"问题；位运算篇的 78「位枚举」是本题的简化版。

### 1255. 得分最高的单词集合（困难）

**题目**：给定单词列表 words、可用字母列表 letters（可重复）和每个字母的分值 score，
用 letters 里的字母拼出 words 的一个子集，每个字母使用次数不能超过它在 letters 里
出现的次数，求能得到的最大总分。

**思路**：

```python
def max_score_words(words, letters, score):
    have = [0] * 26
    for ch in letters:
        have[ord(ch) - 97] += 1

    n = len(words)
    word_count = []
    word_score = []
    for w in words:
        cnt = [0] * 26
        ok = True
        for ch in w:
            idx = ord(ch) - 97
            cnt[idx] += 1
            if cnt[idx] > have[idx]:
                ok = False
        word_count.append(cnt)
        word_score.append(sum(score[ord(ch) - 97] for ch in w) if ok else 0)

    best = 0
    for mask in range(1 << n):
        used = [0] * 26
        total = 0
        ok = True
        for i in range(n):
            if mask >> i & 1:
                for k in range(26):
                    used[k] += word_count[i][k]
                    if used[k] > have[k]:
                        ok = False
                        break
                if not ok:
                    break
                total += word_score[i]
        if ok and total > best:
            best = total
    return best
```

```cpp
int maxScoreWords(std::vector<std::string> &words, std::vector<char> &letters,
                  std::vector<int> &score) {
    int have[26] = {0};
    for (char ch : letters) {
        have[ch - 'a']++;
    }

    int n = static_cast<int>(words.size());
    std::vector<std::vector<int>> wordCount(n, std::vector<int>(26, 0));
    std::vector<int> wordScore(n, 0);
    for (int i = 0; i < n; ++i) {
        bool ok = true;
        for (char ch : words[i]) {
            int idx = ch - 'a';
            wordCount[i][idx]++;
            if (wordCount[i][idx] > have[idx]) {
                ok = false;
            }
            wordScore[i] += score[idx];
        }
        if (!ok) {
            wordScore[i] = 0;
        }
    }

    int best = 0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        int used[26] = {0};
        int total = 0;
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            if (mask >> i & 1) {
                for (int k = 0; k < 26; ++k) {
                    used[k] += wordCount[i][k];
                    if (used[k] > have[k]) {
                        ok = false;
                        break;
                    }
                }
                total += wordScore[i];
            }
        }
        if (ok) {
            best = std::max(best, total);
        }
    }
    return best;
}
```

**为什么这样写**：本题的约束是"每个字母的用量不能超过库存"，不是简单的"集合不重叠"
（因为同一个字母可以在多个单词里出现，只要总量不超）。所以每个单词带一个 26 长度的
计数向量，枚举子集时逐字母累加并与库存比较。预先把"某个字母本身就超库存"的单词判死
并给 0 分，能省掉后续大量无效子集的计算。

n ≤ 14 时，2^n 只有一万多，每次校验 O(n · 26)，完全跑得动。若 n 更大就得上真正的
背包 DP，但本题的数据范围就是为子集枚举准备的。

- **复杂度**：时间 O(2^n · n · 26)，空间 O(n · 26)。
- **易错点**：`word_score` 的累加要在判断"该词本身是否可行"之外单独做——即使某词不可
  单独使用，它的分值也不影响，因为我们只在合法子集里采用它的分；但预判死可以省时间。
  注意 Python 里 `score` 是 0..25 对应 a..z 的数组。
- **相似题**：1239（上题）是"字母不能重复"的版本，本题是"字母不能超量"；691 是"覆盖
  target 的每个位置"，也是子集类问题。

---

## 模式三：状态 = 已覆盖集合

**适用信号**：每一步去"覆盖"一些东西，要问"覆盖全部最少要几步/最少选几个"。

**核心动作**：状态就是"目前已经覆盖了什么"（一个掩码），操作是"再拿一个东西来覆盖"。
因为目标是最少步数或最短路径，通常在状态图上做记忆化搜索或 BFS/DP。

### 691. 贴纸拼词（困难）

**题目**：给定若干贴纸 stickers 和一个目标单词 target。每张贴纸上的字符可以被剪下来
使用（一张贴纸里的每个字符最多用一次），贴纸可以重复购买，求拼出 target 所需的最
少贴纸数；无法拼出返回 -1。

**思路**：

```python
from functools import lru_cache


def min_stickers(stickers, target):
    T = len(target)

    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return 0
        best = float("inf")
        for sticker in stickers:
            cnt = [0] * 26
            for ch in sticker:
                cnt[ord(ch) - 97] += 1
            nxt = remaining
            for i in range(T):
                if nxt >> i & 1:
                    idx = ord(target[i]) - 97
                    if cnt[idx] > 0:
                        cnt[idx] -= 1
                        nxt ^= 1 << i  # 这一位拼好了，清掉
            if nxt != remaining:  # 本贴纸至少有贡献
                best = min(best, 1 + solve(nxt))
        return best

    ans = solve((1 << T) - 1)
    return -1 if ans == float("inf") else ans
```

```cpp
std::vector<std::string> gStickers;
std::string gTarget;
std::unordered_map<int, int> gMemo;

int solve(int remaining) {
    if (remaining == 0) {
        return 0;
    }
    auto it = gMemo.find(remaining);
    if (it != gMemo.end()) {
        return it->second;
    }
    int T = static_cast<int>(gTarget.size());
    int best = 1e9;
    for (const std::string &sticker : gStickers) {
        int cnt[26] = {0};
        for (char ch : sticker) {
            cnt[ch - 'a']++;
        }
        int nxt = remaining;
        for (int i = 0; i < T; ++i) {
            if (nxt >> i & 1) {
                int idx = gTarget[i] - 'a';
                if (cnt[idx] > 0) {
                    cnt[idx]--;
                    nxt ^= 1 << i;
                }
            }
        }
        if (nxt != remaining) {
            best = std::min(best, 1 + solve(nxt));
        }
    }
    return gMemo[remaining] = best;
}

int minStickers(std::vector<std::string> &stickers, std::string target) {
    gStickers = stickers;
    gTarget = target;
    gMemo.clear();
    int T = static_cast<int>(target.size());
    int ans = solve((1 << T) - 1);
    return ans >= 1e9 ? -1 : ans;
}
```

**为什么状态是"还差哪些位置"**：拼到中途，我们只关心"target 的哪些位置还没拼好"，
至多 2^T 种。每次拿一张新贴纸，就尽量去补这些没拼好的位置。用 `remaining` 的二进制位
表示"还没拼好的位置"，`nxt ^= 1 << i` 就是在这一位拼好后把它清掉。每次贴纸让 `remaining`
变小，天然无环，适合记忆化。

**为什么跳过"零贡献"的贴纸**：如果一张贴纸一个字符都补不上（`nxt == remaining`），
买它只会白白 +1，绝不会更优，直接跳过。这条剪枝还能防止状态在原地打转。

- **复杂度**：时间 O(2^T · n · T)（T = len(target)，n = len(stickers)），空间 O(2^T)。
- **易错点**：一张贴纸里的同种字符要按"计数"逐步消耗，不能一见到就全清；Python 用
  `lru_cache` 记得目标长度 ≤ 15；递归深度最多 T 层，不会爆栈。
- **相似题**：1125（下题）把"覆盖 target 位置"换成"覆盖技能集合"，是同一状态定义；
  847 是"覆盖所有节点"的最短路版本；回溯篇的 79 单词搜索是"匹配一个单词"，求和这里是
  "覆盖整个 target"。

### 1125. 最小的必要团队（困难）

**题目**：给定技能列表 req_skills 和每个人的技能 people[i]，选出人数最少的团队，使其
技能的并集覆盖所有 req_skills。

**思路**：

```python
def smallest_sufficient_team(req_skills, people):
    m = len(req_skills)
    skill_id = {skill: i for i, skill in enumerate(req_skills)}

    people_mask = []
    for skills in people:
        mask = 0
        for skill in skills:
            mask |= 1 << skill_id[skill]
        people_mask.append(mask)

    full = (1 << m) - 1
    dp = {0: ()}
    for i, mask in enumerate(people_mask):
        for covered, team in list(dp.items()):
            merged = covered | mask
            if merged == covered:
                continue
            cand = team + (i,)
            if merged not in dp or len(dp[merged]) > len(cand):
                dp[merged] = cand
    return list(dp[full])
```

```cpp
std::vector<int> smallestSufficientTeam(std::vector<std::string> &reqSkills,
                                        std::vector<std::vector<std::string>> &people) {
    int m = static_cast<int>(reqSkills.size());
    std::unordered_map<std::string, int> skillId;
    for (int i = 0; i < m; ++i) {
        skillId[reqSkills[i]] = i;
    }

    int n = static_cast<int>(people.size());
    std::vector<int> peopleMask(n, 0);
    for (int i = 0; i < n; ++i) {
        for (const std::string &skill : people[i]) {
            peopleMask[i] |= 1 << skillId[skill];
        }
    }

    int full = (1 << m) - 1;
    std::vector<std::vector<int>> dp(1 << m);
    std::vector<char> reachable(1 << m, 0);
    dp[0] = {};
    reachable[0] = 1;
    for (int i = 0; i < n; ++i) {
        std::vector<int> masks;
        for (int k = 0; k < (1 << m); ++k) {
            if (reachable[k]) {
                masks.push_back(k);
            }
        }
        for (int covered : masks) {
            int merged = covered | peopleMask[i];
            if (merged == covered) {
                continue;
            }
            std::vector<int> cand = dp[covered];
            cand.push_back(i);
            if (!reachable[merged] || dp[merged].size() > cand.size()) {
                dp[merged] = cand;
                reachable[merged] = 1;
            }
        }
    }
    return dp[full];
}
```

**为什么把"人"放外层**：每个人最多只能进团队一次，这是"每件物品只能选一次"的 0/1 背包。
把人放到外层枚举，内层基于"旧状态快照"扩展，就能保证不会把同一个人重复加进去。状态的
键是"已覆盖的技能集合"，值是"达成该集合的团队名单"。为了最后能输出具体是谁，直接把下标
列表存进状态；如果只求人数，存个整数就行。

**为什么 `merged == covered` 时跳过**：这个人带来的技能已经被覆盖了，加进来只会增加人
数、不会推进任何东西，直接舍弃。

- **复杂度**：时间 O(n · 2^m)，空间 O(2^m · n)（存团队名单）。
- **易错点**：技能编号要先做映射（字符串不能直接当位下标）；`dp` 只保存"人数更少"的
  名单，否则会被更差的方案覆盖；C++ 里同样要先收集所有可达 mask 再更新，避免边遍历
  边扩容。
- **相似题**：691（上题）是"覆盖单词的字符位置"，本题是"覆盖技能集合"，状态定义同构；
  1601 是"枚举请求子集"，也是"选一些元素凑一个平衡"；1434 是"给每个人分配帽子"。

---

## 模式四：枚举请求子集做校验

**适用信号**：有若干独立的选择（比如一批请求/边），要问"最多能同时满足几个"。

**核心动作**：每个选择要么做要么不做，直接枚举所有子集，对每个子集检查是否满足约束，
取最大的可行子集大小。约束往往是"某种平衡"或"某种守恒"。

### 1601. 最多可达成的换楼请求数目（困难）

**题目**：有 n 栋楼，requests[i] = [from, to] 表示一次"从 from 搬到 to"的请求。选出
尽可能多的请求同时满足，使得对每栋楼而言"搬出人数 == 搬入人数"。求最多能同时满足
多少条请求。

**思路**：

```python
def maximum_requests(n, requests):
    best = 0
    for mask in range(1 << len(requests)):
        delta = [0] * n
        count = 0
        for i, (frm, to) in enumerate(requests):
            if mask >> i & 1:
                delta[frm] -= 1
                delta[to] += 1
                count += 1
        if count > best and all(d == 0 for d in delta):
            best = count
    return best
```

```cpp
int maximumRequests(int n, std::vector<std::vector<int>> &requests) {
    int R = static_cast<int>(requests.size());
    int best = 0;
    for (int mask = 0; mask < (1 << R); ++mask) {
        std::vector<int> delta(n, 0);
        int count = 0;
        for (int i = 0; i < R; ++i) {
            if (mask >> i & 1) {
                delta[requests[i][0]]--;
                delta[requests[i][1]]++;
                count++;
            }
        }
        bool ok = true;
        for (int d : delta) {
            if (d != 0) {
                ok = false;
                break;
            }
        }
        if (ok && count > best) {
            best = count;
        }
    }
    return best;
}
```

**为什么约束是"净变化全零"**：一栋楼如果能被满足一批请求，那么搬进和搬出必须一样多，
否则这栋楼的人数会凭空变化。把每条请求看成"from 减一、to 加一"的流量，一组合法请求的
总效果就是每栋楼净流量为零。请求之间没有依赖，所以任何子集都可以独立判断。

R ≤ 16 时 2^R 只有六万多，枚举完全可行。加上 `count > best` 的剪枝（已经不可能超过
当前最优的子集直接跳过校验），更快。

- **复杂度**：时间 O(2^R · R)（校验只需 O(R)，因为 update 与 check 可以一起做），
  空间 O(n)。
- **易错点**：`delta` 每轮都要重置；注意是"from 减、to 加"的方向；`count > best` 的
  判断放在前面可以省掉 `all(d == 0)` 的扫描（Python 里 `all` 本身也是惰性的）。
- **相似题**：本例是"枚举子集 + 守恒校验"最朴素的样子；1125 / 691 是"枚举/扩展子集 +
  覆盖校验"；图论篇的 684 冗余连接是"并查集判环"，与本题的"流量守恒"是完全不同的守恒。

---

## 模式五：状态 = (当前点, 已访问集合)

**适用信号**：在图/网格上走，要求经过所有点，问最短路径；或者把"排列"建模成长度最优。
**核心动作**：在普通 BFS/DP 的状态上，再补一维"已经访问过的点集合"。状态变成
`(当前位置, 访问掩码)`，一次转移走一条边并把新点并入掩码。

### 847. 访问所有节点的最短路径（困难）

**题目**：给定一个连通的无向图（邻接表 graph），求访问所有节点所需的最短路径长度。
可以任意起点、任意终点，节点和边可以重复经过。

**思路**：

```python
from collections import deque


def shortest_path_length(graph):
    n = len(graph)
    full = (1 << n) - 1

    dist = [[-1] * n for _ in range(1 << n)]
    queue = deque()
    for i in range(n):
        mask = 1 << i
        dist[mask][i] = 0
        queue.append((mask, i))

    while queue:
        mask, u = queue.popleft()
        d = dist[mask][u]
        if mask == full:
            return d
        for v in graph[u]:
            nxt = mask | (1 << v)
            if dist[nxt][v] == -1:
                dist[nxt][v] = d + 1
                queue.append((nxt, v))
    return -1
```

```cpp
int shortestPathLength(std::vector<std::vector<int>> &graph) {
    int n = static_cast<int>(graph.size());
    int full = (1 << n) - 1;

    std::vector<std::vector<int>> dist(1 << n, std::vector<int>(n, -1));
    std::queue<std::pair<int, int>> q;
    for (int i = 0; i < n; ++i) {
        int mask = 1 << i;
        dist[mask][i] = 0;
        q.push({mask, i});
    }

    while (!q.empty()) {
        auto [mask, u] = q.front();
        q.pop();
        int d = dist[mask][u];
        if (mask == full) {
            return d;
        }
        for (int v : graph[u]) {
            int nxt = mask | (1 << v);
            if (dist[nxt][v] == -1) {
                dist[nxt][v] = d + 1;
                q.push({nxt, v});
            }
        }
    }
    return -1;
}
```

**为什么用 BFS 而不是普通 Dijkstra**：每条边代价都是 1，边的条数就是步数，所以按层
扩展的 BFS 天然给出最短路。这里的状态是 `(mask, u)`，BFS 在"状态图"上跑，第一次到达
`mask == full` 的状态，步数就是答案。

**为什么所有起点一起入队**：题目允许任意起点，把每个 `(1 << i, i)` 都作为距离 0 的
源点塞进队列（多源 BFS），一次遍历就能算出"从任意点出发覆盖全图"的最短长度，省掉
对每个起点各跑一遍。

- **复杂度**：时间 O(2^n · (n + m))（状态数 2^n · n，每个状态扩展其出边），空间 O(2^n · n)。
- **易错点**：`dist` 是二维的，`dist[mask][u]`，别只开一维；每个节点可以重复经过，
  所以不能像普通图那样只标记"访问过的节点"，必须把"访问集合"写进状态；n=1 时答案 0。
- **相似题**：943（下题）是它的"带权 + 要求排列顺序"版本（TSP）；最短路径篇的 1091
  是普通网格 BFS；本题的"状态多一维集合"技巧，在旅行商、状压 DP 里是核心。

### 943. 最短超级串（困难）

**题目**：给定字符串数组 words，返回一个最短的字符串，使 words 里每个词都是它的子串。

**思路**：

```python
def shortest_superstring(words):
    # 去重 + 去掉是别人子串的词
    unique = []
    for w in words:
        if w not in unique:
            unique.append(w)
    words = [w for w in unique
             if not any(w != other and w in other for other in unique)]
    k = len(words)
    if k == 0:
        return ""

    overlap = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            a, b = words[i], words[j]
            for length in range(min(len(a), len(b)), 0, -1):
                if a[-length:] == b[:length]:
                    overlap[i][j] = length
                    break

    full = (1 << k) - 1
    INF = float("inf")
    dp = [[INF] * k for _ in range(1 << k)]
    parent = [[-1] * k for _ in range(1 << k)]
    for i in range(k):
        dp[1 << i][i] = len(words[i])

    for mask in range(1 << k):
        for last in range(k):
            if dp[mask][last] == INF:
                continue
            for nxt in range(k):
                if mask >> nxt & 1:
                    continue
                nm = mask | (1 << nxt)
                cand = dp[mask][last] + len(words[nxt]) - overlap[last][nxt]
                if cand < dp[nm][nxt]:
                    dp[nm][nxt] = cand
                    parent[nm][nxt] = last

    best_len = INF
    best_last = -1
    for i in range(k):
        if dp[full][i] < best_len:
            best_len = dp[full][i]
            best_last = i

    # 回推排列
    order = []
    mask, last = full, best_last
    while last != -1:
        order.append(last)
        prev = parent[mask][last]
        mask ^= 1 << last
        last = prev
    order.reverse()

    ans = words[order[0]]
    for t in range(1, len(order)):
        i, j = order[t - 1], order[t]
        ans += words[j][overlap[i][j]:]
    return ans
```

```cpp
std::string shortestSuperstring(std::vector<std::string> words) {
    std::vector<std::string> unique;
    for (const std::string &w : words) {
        if (std::find(unique.begin(), unique.end(), w) == unique.end()) {
            unique.push_back(w);
        }
    }
    words.clear();
    for (const std::string &w : unique) {
        bool redundant = false;
        for (const std::string &other : unique) {
            if (w != other && other.find(w) != std::string::npos) {
                redundant = true;
                break;
            }
        }
        if (!redundant) {
            words.push_back(w);
        }
    }

    int k = static_cast<int>(words.size());
    if (k == 0) {
        return "";
    }
    std::vector<std::vector<int>> overlap(k, std::vector<int>(k, 0));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i == j) {
                continue;
            }
            const std::string &a = words[i];
            const std::string &b = words[j];
            int limit = std::min(a.size(), b.size());
            for (int length = limit; length > 0; --length) {
                if (a.substr(a.size() - length) == b.substr(0, length)) {
                    overlap[i][j] = length;
                    break;
                }
            }
        }
    }

    int full = (1 << k) - 1;
    const int INF = 1e9;
    std::vector<std::vector<int>> dp(1 << k, std::vector<int>(k, INF));
    std::vector<std::vector<int>> parent(1 << k, std::vector<int>(k, -1));
    for (int i = 0; i < k; ++i) {
        dp[1 << i][i] = static_cast<int>(words[i].size());
    }

    for (int mask = 0; mask < (1 << k); ++mask) {
        for (int last = 0; last < k; ++last) {
            if (dp[mask][last] == INF) {
                continue;
            }
            for (int nxt = 0; nxt < k; ++nxt) {
                if (mask >> nxt & 1) {
                    continue;
                }
                int nm = mask | (1 << nxt);
                int cand = dp[mask][last] + static_cast<int>(words[nxt].size()) - overlap[last][nxt];
                if (cand < dp[nm][nxt]) {
                    dp[nm][nxt] = cand;
                    parent[nm][nxt] = last;
                }
            }
        }
    }

    int bestLen = INF;
    int bestLast = -1;
    for (int i = 0; i < k; ++i) {
        if (dp[full][i] < bestLen) {
            bestLen = dp[full][i];
            bestLast = i;
        }
    }

    std::vector<int> order;
    int mask = full;
    int last = bestLast;
    while (last != -1) {
        order.push_back(last);
        int prev = parent[mask][last];
        mask ^= 1 << last;
        last = prev;
    }
    std::reverse(order.begin(), order.end());

    std::string ans = words[order[0]];
    for (int t = 1; t < static_cast<int>(order.size()); ++t) {
        int i = order[t - 1];
        int j = order[t];
        ans += words[j].substr(overlap[i][j]);
    }
    return ans;
}
```

**为什么是 TSP**：答案一定是由这些词按某个顺序拼出来的，相邻两个词之间能"咬合"
（前一个的后缀 = 后一个的前缀）。要让超级串最短，就是找一个排列，使"所有咬合节省的
字符"最多。把每个词当成一个城市，从 i 走到 j 的代价是 `len(j) - overlap[i][j]`，
问题就是旅行商问题。

**为什么状态要带 last**：只记"用了哪些词"不够——下一个词能咬合多少，取决于上一个词
是谁。所以 `dp[mask][last]` 记的是"已用集合为 mask、最后一个词是 last"的最短长度。
转移就是"在末尾再挂一个没用过的词"。

**为什么要 parent**：本题要输出具体的串，只算长度不够。用 `parent[mask][last]` 记下
最优转移的前驱，最后从 `(full, best_last)` 往回走，就能还原出词的顺序。

- **复杂度**：时间 O(2^k · k^2)，空间 O(2^k · k)。k 是去重去子串后的词数（通常 ≤ 12）。
- **易错点**：先做两项预处理——去重、去掉是别人子串的词，否则既浪费状态又可能算错；
  `overlap` 要取"最长重合"；回推排列时 `mask ^= 1 << last` 在确定 last 在集合里时才安全。
- **相似题**：847（上题）是它的无权版本；动态规划篇的 312 戳气球、博弈论篇的状态压缩
  题都用到"状态 = 已处理集合"；本题的 `parent` 回推是"输出方案"类 DP 的通用手法。

---

## 模式六：状态 = 已分配对象的集合

**适用信号**：把若干"资源"分配给若干"对象"，每个对象只能拿一个、每个资源最多用一次。

**核心动作**：把"对象"（或"资源"）压成一个掩码当状态，把另一侧放到外层枚举，用 0/1
背包式的转移统计方案数或求最优。

### 1434. 每个人戴不同帽子的方案数（困难）

**题目**：有 n 个人和若干顶帽子（编号 1..40）。hats[i] 是第 i 个人喜欢的所有帽子编号。
给每个人戴一顶喜欢的帽子，要求任意两人帽子不同，求方案数（对 1e9+7 取模）。

**思路**：

```python
def number_ways(hats):
    MOD = 10 ** 9 + 7
    n = len(hats)

    # 每顶帽子被哪些人喜欢（先对每个人的列表去重）
    likers = [[] for _ in range(41)]
    for i in range(n):
        for h in set(hats[i]):
            likers[h].append(i)

    dp = [0] * (1 << n)
    dp[0] = 1
    for h in range(1, 41):
        if not likers[h]:
            continue
        nxt = dp[:]  # 这顶帽子不分配，方案数原样保留
        for mask in range(1 << n):
            if dp[mask] == 0:
                continue
            for i in likers[h]:
                if not (mask >> i & 1):
                    nxt[mask | (1 << i)] = (nxt[mask | (1 << i)] + dp[mask]) % MOD
        dp = nxt
    return dp[(1 << n) - 1]
```

```cpp
int numberWays(std::vector<std::vector<int>> &hats) {
    const int MOD = 1000000007;
    int n = static_cast<int>(hats.size());

    std::vector<std::vector<int>> likers(41);
    for (int i = 0; i < n; ++i) {
        std::set<int> uniq(hats[i].begin(), hats[i].end());
        for (int h : uniq) {
            likers[h].push_back(i);
        }
    }

    std::vector<long long> dp(1 << n, 0);
    dp[0] = 1;
    for (int h = 1; h <= 40; ++h) {
        if (likers[h].empty()) {
            continue;
        }
        std::vector<long long> nxt = dp;
        for (int mask = 0; mask < (1 << n); ++mask) {
            if (dp[mask] == 0) {
                continue;
            }
            for (int i : likers[h]) {
                if (!(mask >> i & 1)) {
                    nxt[mask | (1 << i)] = (nxt[mask | (1 << i)] + dp[mask]) % MOD;
                }
            }
        }
        dp = nxt;
    }
    return static_cast<int>(dp[(1 << n) - 1]);
}
```

**为什么帽子放外层、人用掩码**：一顶帽子只能给一个人，这正是"每件物品只能用一次"。
把 40 顶帽子逐个作为"物品"枚举，状态记录"哪些人已经戴好帽子"，就避免了同一顶帽子
被两个人戴。每顶帽子有两种去向：不分配（继承旧方案数），或分配给一个还空着且喜欢它
的人（把方案数加过去）。

**为什么用 `nxt` 而不是原地更新**：如果直接在 `dp` 上改，同一顶帽子可能在一次转移里
被"连环"分配（先给甲、再基于新状态给乙），违反"一顶只给一人"。用旧数组 `dp` 读、新
数组 `nxt` 写，就能保证每顶帽子只分配一次。这是 0/1 背包"倒序/快照"的又一种体现。

- **复杂度**：时间 O(40 · 2^n · n)，空间 O(2^n)。
- **易错点**：输入里同一个人可能重复列出同一顶帽子，先对 `hats[i]` 去重，否则方案数会
  翻倍；转移读 `dp` 写 `nxt`，最后再 `dp = nxt`；取模别漏。
- **相似题**：1125 是"选人覆盖技能"，本题是"分帽子给人"，都是掩码状态 + 0/1 转移；
  随机化篇的 528 / 497 也在"按某种权重分配"，但那是抽样不是计数。

---

## 规律总结

1. **状态压缩 = 用整数的二进制位表示集合**。判断元素在不在用 `mask >> i & 1`，加入用
   `mask | (1 << i)`，判冲突用 `mask1 & mask2 == 0`，装满用 `mask == (1 << n) - 1`。
   把这几个式子背下来，状态压缩就成功了一半。

2. **先问"元素有几个、每个是不是只有两种状态"**。只有"在/不在"，且 n 小（一般 ≤ 20），
   才适合压缩；否则状态数 2^n 会爆。题目里 n ≤ 12/16/20 的小上界，就是最好的提示。

3. **枚举所有子集就是 `for mask in range(1 << n)`**。78、784、1601 都是这一句的直接
   套用。真正需要决策的元素可能只是其中一部分（784 只对字母），先把它们挑出来，mask
   的有效位数才不会被浪费。

4. **"选一个子集 + 物品间有冲突/约束" → 掩码当校验器**。1239 用 `m1 & m2 == 0` 判
   字母冲突，1255 用计数向量判超量，1125 用 `|` 累加覆盖。掩码负责记录"用了什么"，
   冲突判断就是一次位运算。

5. **"覆盖一个集合，求最少步数/最少个数" → 状态是已覆盖集合**。691（覆盖字符位置）、
   1125（覆盖技能）、847（覆盖节点）都是这个套路。每次操作让状态"变大"，天然无环，
   适合记忆化搜索或 BFS。

6. **"在图/排列上要求经过所有点" → 状态补一维访问掩码**。847 是 `(当前点, 访问集合)`
   的 BFS，943 是 `(最后位置, 已用集合)` 的 TSP DP。这一招是旅行商、状压 DP 的通用骨架。

7. **外层枚举"只能用一次"的物品，状态存"另一侧"的集合**。1434 帽子在外层、人做状态；
   1125 人放外层、技能做状态。配合"读旧写新"（快照/`nxt` 数组），就实现了"每件物品
   最多用一次"的 0/1 语义。

8. **要输出方案就多存一个 parent**。943 用 `parent[mask][last]` 回推词的顺序；只在求
   最优值时才不用。凡是"要构造出具体答案"的 DP，都预留一个"我从哪来"的数组。

9. **0/1 扩展必须基于旧状态**。Python 遍历 `list(dp.items())`，C++ 先拷一份快照或另开
   `nxt` 数组，否则会在同一次迭代里复用刚加进去的新状态，悄悄变成"物品可用多次"。

10. **小上界不等于暴力一定过，该剪枝就剪枝**。691 跳过"零贡献"的贴纸，1601 先比
    `count > best`，1125 跳过"已被覆盖"的人——这些剪枝常常能把常数降一个量级。

11. **与其它篇的联系**：掩码本身是第 16 篇「位运算」的核心工具；`dp[mask]` 的状态扩展
    是第 13 篇「动态规划」在集合上的应用；691 / 1125 的"逐层覆盖、记忆化"与第 11 篇
    「回溯」的剪枝同源；847 / 943 的 `(位置, 集合)` 状态与第 22 篇「最短路径」的"把
    限制做成状态维度"是同一个思想。状态压缩可以说是"位运算 + DP + 图搜索"三者的交汇。
