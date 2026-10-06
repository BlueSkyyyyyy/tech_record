# 前缀树：用「共享前缀」组织字符串

前缀树（Trie，也叫字典树）是一棵专门用来存字符串的树。它的思想很朴素：**把字符串按字符
一位一位拆开，公共的前缀只存一份**。比如 `apple` 和 `app` 共享前三个字符，在树里它们就
走同一条路，到 `app` 处先记一个「到此是一个完整单词」，再继续往下长出 `l`、`e`。树根到某个
节点的路径，就是这些节点共同拥有的那段前缀。

这种结构最大的价值，是把「字符串的集合」从「一堆彼此独立的值」变成了「一棵能按前缀剪枝
的树」。因此只要题目里出现**前缀、补全、通配符、词根替换、按前缀聚合**这些信号，就应该先想
前缀树。本篇先用 208 立起最基础的节点与三个操作，再用 211 加上通配符、用 212 把它和网格
回溯拼在一起，最后用 648 与 677 展示它作为「索引结构」的两种常见用法。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：前缀树的基本骨架 | 208. 实现 Trie（前缀树） | 中等 |
| 模式二：带通配符的搜索 | 211. 添加与搜索单词 - 数据结构设计 | 中等 |
| 模式三：前缀树 + 网格回溯 | 212. 单词搜索 II | 困难 |
| 模式四：找最短词根 | 648. 单词替换 | 中等 |
| 模式五：维护前缀聚合值 | 677. 键值映射 | 中等 |

读这几题时抓住一条主线：**前缀树节点不是存「一个单词」，而是存「走到这里的一条前缀」**。
理解了这一点，节点的 `children`（下一个字符到子节点）、`is_end`（是否在此结束），以及
「沿前缀走一遍就能回答的问题」，都会顺理成章。

---

## 模式一：前缀树的基本骨架

**适用信号**：需要在字符串集合上反复做「插入」「是否出现过某个完整单词」「是否存在以某
前缀开头的单词」这三类查询。

**核心结构**：每个节点有两个信息——`children`（字典或数组，键是下一个字符，值是对应子
节点）和 `is_end`（布尔量，标记「有一个插入的单词恰好在这里结束」）。整棵树的根表示空
前缀。

### 208. 实现 Trie（前缀树）（中等）

**题目**：实现一棵前缀树，支持 `insert(word)`、`search(word)`、`starts_with(prefix)`。
其中 `search` 要求完整匹配，`starts_with` 只要求前缀存在。

**思路**：三个操作都从根出发，按字符逐层往下走。`insert` 遇到不存在的字符就新建节点，
走完后把末节点的 `is_end` 置真；`search` 与 `starts_with` 都沿着字符往下走，区别只在
「走完之后」——`search` 要求当前节点 `is_end` 为真，`starts_with` 只要节点存在即可。
所以可以把「沿前缀走一遍」抽成一个辅助函数 `_find`，让两个查询共用。

**为什么用树而不是集合**：集合只能回答「整个单词在不在」，答不了「有没有单词以某前缀
开头」。前缀树把公共前缀共享在一棵树上，顺着字符走一遍就同时经过了沿途的每个前缀，
于是前缀查询成了「路径存不存在」这个自然的问题。查询长度 L 的串只需 O(L)，与树里存了
多少单词无关。

**代码**（完整可运行版见 `src/trie/implement_trie.py` / `.cpp`）：

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True

    def _find(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None
```

```cpp
class Trie {
  public:
    Trie() { children_.fill(nullptr); }
    ~Trie() {
        for (Trie *child : children_) delete child;
    }

    Trie(const Trie &) = delete;
    Trie &operator=(const Trie &) = delete;

    void insert(const std::string &word) {
        Trie *node = this;
        for (char ch : word) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new Trie();
            node = node->children_[i];
        }
        node->isEnd_ = true;
    }

    bool search(const std::string &word) const {
        const Trie *node = find(word);
        return node != nullptr && node->isEnd_;
    }

    bool startsWith(const std::string &prefix) const {
        return find(prefix) != nullptr;
    }

  private:
    std::array<Trie *, 26> children_;
    bool isEnd_ = false;

    const Trie *find(const std::string &prefix) const {
        const Trie *node = this;
        for (char ch : prefix) {
            int i = ch - 'a';
            if (!node->children_[i]) return nullptr;
            node = node->children_[i];
        }
        return node;
    }
};
```

- **复杂度**：三个操作都是 O(L)，L 为单词长度；空间 O(总字符数)。
- **易错点**：`search` 不能只判断「路径存在」，必须看末节点的 `is_end`——否则插入了
  `apple` 后 `search("app")` 会误判为真；根节点也要有 `is_end`，才能支持空串（若题目
  允许）。C++ 里用 26 个指针的数组比 `unordered_map` 更快，但只适用于小写字母表。
- **相似题**：211 在搜索时加入通配符；648 用它找最短词根；677 在每个节点上再挂一个
  聚合值。这四题共用同一副骨架。

---

## 模式二：带通配符的搜索

**适用信号**：查询里含有能匹配「任意一个字符」的通配符（如 `.`），需要判断是否存在
某个已插入的单词与这个模式完全匹配。

**核心动作**：普通字符照常沿边走；遇到通配符时，当前节点的**所有**子节点都值得一试，
于是把查询写成回溯。

### 211. 添加与搜索单词 - 数据结构设计（中等）

**题目**：设计一个结构，支持 `add_word(word)` 添加单词，以及 `search(word)` 搜索；
`word` 中可能含 `.`，它可匹配任意一个字母。

**思路**：树和插入逻辑与 208 一样。搜索时从根开始逐字符匹配：当前字符是普通字母，就沿
对应子节点往下；是 `.` 时，对当前节点的每个子节点递归匹配剩余部分，任意一条分支成功
即返回真。当模式串走完时，判断当前节点是否 `is_end`。

**为什么通配符必须回溯**：一个 `.` 把「唯一一条路径」分叉成「所有可能的孩子」多条路径，
只有走到底才能知道哪条能匹配成功，贪心地选一个是不行的。普通字符没有分支，所以继续往
下即可。也正因如此，含通配符的查询最坏会退化成枚举（形如 `"..."` 且树很宽时），这是
问题本身的要求，不是实现缺陷。

**代码**（完整可运行版见 `src/trie/add_and_search_words.py` / `.cpp`）：

```python
class WordDictionary:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def add_word(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = WordDictionary()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        return self._dfs(self, word, 0)

    def _dfs(self, node, word, i):
        if i == len(word):
            return node.is_end
        ch = word[i]
        if ch == '.':
            for child in node.children.values():
                if self._dfs(child, word, i + 1):
                    return True
            return False
        if ch not in node.children:
            return False
        return self._dfs(node.children[ch], word, i + 1)
```

```cpp
class WordDictionary {
  public:
    WordDictionary() { children_.fill(nullptr); }
    ~WordDictionary() {
        for (WordDictionary *child : children_) delete child;
    }

    WordDictionary(const WordDictionary &) = delete;
    WordDictionary &operator=(const WordDictionary &) = delete;

    void addWord(const std::string &word) {
        WordDictionary *node = this;
        for (char ch : word) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new WordDictionary();
            node = node->children_[i];
        }
        node->isEnd_ = true;
    }

    bool search(const std::string &word) const { return dfs(this, word, 0); }

  private:
    std::array<WordDictionary *, 26> children_;
    bool isEnd_ = false;

    bool dfs(const WordDictionary *node, const std::string &word, int i) const {
        if (i == static_cast<int>(word.size())) return node->isEnd_;
        char ch = word[i];
        if (ch == '.') {
            for (const WordDictionary *child : node->children_) {
                if (child && dfs(child, word, i + 1)) return true;
            }
            return false;
        }
        const WordDictionary *child = node->children_[ch - 'a'];
        if (!child) return false;
        return dfs(child, word, i + 1);
    }
};
```

- **复杂度**：`add_word` 为 O(L)；`search` 普通查询 O(L)，含通配符时最坏 O(26^L)。
- **易错点**：递归出口是「模式串走完」而非「节点为空」，否则无法区分「前缀存在」和
  「完整单词存在」；`.` 分支要遍历所有孩子，别只取第一个；C++ 中 26 个孩子里有空指针，
  遍历时要先判空。
- **相似题**：208 是它的无通配符版本；212 把它搬到网格上；正则匹配类题目（如 10 正则
  表达式匹配）也用到「通配符分叉 + 回溯」的同一招。

---

## 模式三：前缀树 + 网格回溯

**适用信号**：在字符网格里找出「给定单词列表」中所有能走出来的单词。若对每个单词单独
搜索，会有大量重复；此时用前缀树把单词集合组织起来做统一剪枝。

**核心动作**：把单词全部插进前缀树，然后从网格每个格子出发只做一次 DFS，边走边沿树
下行，走不动就剪枝，走到单词结尾就收答案。

### 212. 单词搜索 II（困难）

**题目**：给定字符网格 `board` 和单词列表 `words`，找出所有能在网格中由相邻格子
（上下左右、不重复使用）依次连接而成的单词。

**思路**：先建树，再枚举起点。DFS 的参数里带上前缀树的当前节点：如果当前格子的字母
在节点的孩子里，就进入那个孩子继续搜；若该孩子标记着某个单词结尾，就收集它。同一
单词可能被多条路径命中，找到后把结尾标记清掉即可去重。遍历时把格子临时改成一个非
字母字符表示「已访问」，回溯时还原。

**为什么前缀树能把「多次搜索」压成「一次搜索」**：所有单词共享前缀，网格 DFS 选了一个
字母后，只要它还在「某些待搜单词的前缀」里就继续，否则整条分支都不可能凑出任何单词，
立刻砍掉。前缀树用 O(1) 判断「当前前缀是否还在树上」，把大量无效分支提前截断。反过
来说，如果没有共享前缀，比如单词列表全是互不相干的长串，前缀树的收益就有限，这也是
它最坏复杂度仍为 O(m·n·4^L) 的原因。

**代码**（完整可运行版见 `src/trie/word_search_ii.py` / `.cpp`）：

```python
def find_words(board, words):
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = word

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        ch = board[r][c]
        nxt = node.get(ch)
        if nxt is None:
            return
        word = nxt.get("#")
        if word is not None:
            result.append(word)
            nxt["#"] = None  # 置空，避免同一单词被重复收集
        board[r][c] = ""  # 标记已访问
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] in nxt:
                dfs(nr, nc, nxt)
        board[r][c] = ch  # 还原

    for r in range(rows):
        for c in range(cols):
            if board[r][c] in trie:
                dfs(r, c, trie)
    return result
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 26> child;
    std::string word;
    TrieNode() { child.fill(nullptr); }
};

class WordSearchII {
  public:
    std::vector<std::string> findWords(std::vector<std::vector<char>> &board,
                                       const std::vector<std::string> &words) {
        TrieNode *root = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root;
            for (char ch : word) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->word = word;
        }

        result_.clear();
        rows_ = static_cast<int>(board.size());
        cols_ = static_cast<int>(board[0].size());
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c) dfs(board, r, c, root);
        return result_;
    }

  private:
    std::vector<std::string> result_;
    int rows_ = 0;
    int cols_ = 0;

    void dfs(std::vector<std::vector<char>> &board, int r, int c, TrieNode *node) {
        char ch = board[r][c];
        TrieNode *nxt = node->child[ch - 'a'];
        if (!nxt) return;
        if (!nxt->word.empty()) {
            result_.push_back(nxt->word);
            nxt->word.clear();  // 置空，避免同一单词被重复收集
        }
        board[r][c] = '#';
        const int dr[4] = {1, -1, 0, 0};
        const int dc[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && board[nr][nc] != '#')
                dfs(board, nr, nc, nxt);
        }
        board[r][c] = ch;
    }
};
```

- **复杂度**：建树 O(总字符数)；搜索最坏 O(m·n·4^L)，L 是最长单词长度，实际因前缀
  剪枝远小于此；空间 O(总字符数 + L)（不计结果）。
- **易错点**：找到单词后必须把结尾标记清空，否则同一单词会被多条路径重复收集；
  标记与还原要成对，`board[r][c]` 走完必须恢复；先判断「字母是否在孩子的键里」再递归，
  不要进了格子才发现走不通。
- **相似题**：79 单词搜索（单个单词版，`backtracking` 篇）；208 / 211 是它的前缀树
  基础；1268 搜索推荐系统在同一棵树上按前缀返回 Top-K。

---

## 模式四：找最短词根

**适用信号**：给定一批「前缀」，要把一个字符串替换成它匹配到的最短前缀。

**核心动作**：把前缀集合建成前缀树，沿字符串下行，遇到的第一个 `is_end` 就是最短匹配。

### 648. 单词替换（中等）

**题目**：给定词典 `dictionary`（若干词根）和用空格分隔的句子 `sentence`，把每个单词
替换成能作为它前缀的最短词根；没有对应词根的保持原样。

**思路**：先把词根建树。处理每个单词时从根往下走，边走边把已走过的字符记成 `built`；
一旦当前节点是某个词根的结尾，就说明找到了最短词根，直接停下；中途走不通则说明没有
可用词根，保留原单词。

**为什么「第一次遇到结尾」就是最短**：词根在前缀树上是从根往下的路径，结尾出现得越早，
路径越短。所以顺着走，第一个 `is_end` 天然就是最短的那个，不用收集所有匹配再比较长度。
这正是前缀结构「顺序即长度」的好处。

**代码**（完整可运行版见 `src/trie/replace_words.py` / `.cpp`）：

```python
def replace_words(dictionary, sentence):
    trie = {}
    for root in dictionary:
        node = trie
        for ch in root:
            node = node.setdefault(ch, {})
        node["#"] = True

    result = []
    for word in sentence.split():
        node = trie
        built = []
        for ch in word:
            if "#" in node or ch not in node:
                break
            node = node[ch]
            built.append(ch)
        if node.get("#"):
            result.append("".join(built))
        else:
            result.append(word)
    return " ".join(result)
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 26> child;
    bool isEnd = false;
    TrieNode() { child.fill(nullptr); }
};

std::string replaceWords(const std::vector<std::string> &dictionary, const std::string &sentence) {
    TrieNode *root = new TrieNode();
    for (const std::string &rootWord : dictionary) {
        TrieNode *node = root;
        for (char ch : rootWord) {
            int i = ch - 'a';
            if (!node->child[i]) node->child[i] = new TrieNode();
            node = node->child[i];
        }
        node->isEnd = true;
    }

    std::string result;
    std::istringstream iss(sentence);
    std::string word;
    bool first = true;
    while (iss >> word) {
        if (!first) result += ' ';
        first = false;
        TrieNode *node = root;
        std::string built;
        for (char ch : word) {
            if (node->isEnd) break;  // 已到某个词根，最短词根即它
            int i = ch - 'a';
            if (!node->child[i]) break;
            node = node->child[i];
            built += ch;
        }
        result += node->isEnd ? built : word;
    }
    return result;
}
```

- **复杂度**：建树 O(总词根长度)；处理句子 O(句子总长度)；空间 O(总词根长度)。
- **易错点**：判断「已到词根」要在读取每个字符**之前**做，否则会多匹配一个字符；
  `built` 只记录真正走过的字符，走不通时要丢弃；C++ 按空格切词用 `istringstream`
  最省事，拼接空格要注意首词不加。
- **相似题**：677 用前缀树维护聚合；208 / 211 是基础；14 最长公共前缀（`string` 篇）
  是「公共前缀」的另一类问题，但用纵向扫描而非树。

---

## 模式五：维护前缀聚合值

**适用信号**：查询「所有以某前缀开头的键的某种聚合值（如求和）」，同时键会被更新或
覆盖。

**核心动作**：在节点上额外存一个聚合量（这里是 `total`），插入时把「值的变化量」沿
路径累加下去。

### 677. 键值映射（中等）

**题目**：设计一个 map，支持 `insert(key, val)`（覆盖旧值）与 `sum(prefix)`（返回所有
以 `prefix` 为前缀的键的值之和）。

**思路**：每个节点维护 `total`，表示「经过这个节点的所有键的值之和」。插入时先求出增量
`delta = 新值 - 旧值`（旧值不存在按 0 算），再沿键的字符往下走，每经过一个节点就给它
的 `total` 加上 `delta`，最后在末节点记下最新值。查询时走到前缀末节点，返回它的
`total`。

**为什么用增量而不是重算**：覆盖旧值时，要先把旧值从路径上减掉、再把新值加上。若直接
把整条路径都加上新值，旧值就被重复累加了。用 `delta` 一次修正，插入仍是 O(L)。这是
「覆盖型更新」的通用技巧：**先求差，再把差施加到所有受影响的聚合量上**。

**代码**（完整可运行版见 `src/trie/map_sum_pairs.py` / `.cpp`）：

```python
class MapSum:
    def __init__(self):
        self.children = {}
        self.total = 0
        self.value = 0

    def insert(self, key, val):
        delta = val - self._get(key)
        node = self
        for ch in key:
            if ch not in node.children:
                node.children[ch] = MapSum()
            node = node.children[ch]
            node.total += delta
        node.value = val

    def sum(self, prefix):
        node = self
        for ch in prefix:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.total

    def _get(self, key):
        node = self
        for ch in key:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.value
```

```cpp
class MapSum {
  public:
    MapSum() { children_.fill(nullptr); }
    ~MapSum() {
        for (MapSum *child : children_) delete child;
    }

    MapSum(const MapSum &) = delete;
    MapSum &operator=(const MapSum &) = delete;

    void insert(const std::string &key, int val) {
        int delta = val - getValue(key);
        MapSum *node = this;
        for (char ch : key) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new MapSum();
            node = node->children_[i];
            node->total_ += delta;
        }
        node->value_ = val;
    }

    int sum(const std::string &prefix) const {
        const MapSum *node = this;
        for (char ch : prefix) {
            int i = ch - 'a';
            if (!node->children_[i]) return 0;
            node = node->children_[i];
        }
        return node->total_;
    }

  private:
    std::array<MapSum *, 26> children_;
    int total_ = 0;
    int value_ = 0;

    int getValue(const std::string &key) const {
        const MapSum *node = this;
        for (char ch : key) {
            int i = ch - 'a';
            if (!node->children_[i]) return 0;
            node = node->children_[i];
        }
        return node->value_;
    }
};
```

- **复杂度**：`insert` 与 `sum` 均为 O(L)；空间 O(总字符数)。
- **易错点**：必须用增量 `delta` 更新路径，不能用全新值直接累加；`sum` 在前缀不存在
  时返回 0；`get` 旧值返回 0 只对「值非负且新键」成立，若题目允许 `val` 为 0 要用别的
  标记区分「不存在」。
- **相似题**：208 是它的无聚合版本；560 和为 K 的子数组、437 路径总和 III（`prefix-sum`
  篇）也是「把前缀上的信息聚合成答案」；745 前缀和后缀搜索把它推广到双向前缀。

---

## 规律总结

1. **前缀树节点描述的是「一条前缀」，不是一个单词**：`children` 负责分叉，`is_end`
   负责标记「到此恰好有一个完整单词」。几乎所有前缀树变体，都是在这两个字段上做文章。

2. **三个基础操作只差「走完之后做什么」**：`insert` / `search` / `starts_with` 都沿
   字符下行，因此能把「走路径」抽成公共辅助函数（208 的 `_find`）。写新题时先想清楚
   「走到前缀末尾后，是判断 `is_end`、还是返回某个聚合量、还是继续递归」。

3. **通配符 = 分叉 = 回溯**：普通字符分支唯一，通配符把分支变成一个集合。遇到 `.`
   这类可匹配任意字符的模式，就把搜索写成回溯，穷举所有可能的孩子，任意成功即真。
   代价是最坏指数级，但这是问题本身决定的。

4. **前缀树最擅长「给一批字符串做统一剪枝」**：212 把「对每个单词各搜一遍网格」变成
   「所有单词共享一次网格 DFS」，靠的就是「当前前缀是否还在树上」。凡是「一组字符串
   在同一个结构里寻找」的题，都应先考虑用前缀树合并搜索。

5. **顺序即长度**：前缀树上从根往下的路径天然按长度排列，所以 648 沿路遇到的第一个
   `is_end` 就是最短词根，不需要额外比较。只要题目问「最短 / 最先匹配的前缀」，顺着
   走就能拿到。

6. **覆盖型更新用「增量」而不是「重算」**：677 插入时先求 `delta = 新值 - 旧值`，再把
   `delta` 施加到路径上所有节点。这样一次插入仍是 O(L)，也不会把旧值重复累加。这个
   先求差再统一施加的思路，在别处的「增量维护」问题里同样通用。

7. **节点的额外字段 = 额外能力**：`is_end` 只回答「是不是单词」，再加上 `total` 就能
   回答「前缀和」，再加上 `word` 就能直接返回命中的单词。需要什么查询，就在节点上挂
   什么信息，让查询在「走到前缀」的过程中顺手得到。

8. **字母表大小决定存储方式**：小写字母用长度 26 的数组，O(1) 定位、常数小；字符集
   很大或稀疏时用哈希表，省空间但常数略大。C++ 用数组要记得把指针初始化为 `nullptr`，
   且若用裸指针要注意释放，避免内存泄漏。

9. **前缀树与其它结构的交汇**：它和哈希表都能存字符串，但哈希表答不了前缀问题；
   它和回溯天然契合（212），和聚合维护天然契合（677）。判断该不该用前缀树，就问一句：
   **问题里有没有「前缀」二字，或者能不能被翻译成前缀问题。**
