# 前缀树：用「共享前缀」组织字符串

前缀树（Trie，也叫字典树）是一棵专门用来存字符串的树。它的思想很朴素：**把字符串按字符
一位一位拆开，公共的前缀只存一份**。比如 `apple` 和 `app` 共享前三个字符，在树里它们就
走同一条路，到 `app` 处先记一个「到此是一个完整单词」，再继续往下长出 `l`、`e`。树根到某个
节点的路径，就是这些节点共同拥有的那段前缀。

这种结构最大的价值，是把「字符串的集合」从「一堆彼此独立的值」变成了「一棵能按前缀剪枝
的树」。因此只要题目里出现**前缀、补全、通配符、词根替换、按前缀聚合**这些信号，就应该先想
前缀树。本篇先用 208 立起最基础的节点与三个操作，再用 211 加上通配符、用 212 把它和网格
回溯拼在一起，再用 648 与 677 展示它作为「索引结构」的两种常见用法，最后用 720、1268、
1032、745、421 把「沿前缀找最长链、节点挂 Top-K、反向匹配后缀、同时匹配前后缀、按位贪心」
这几副新面孔补齐。

| 模式 | 题目 | 难度 |
|---|---|---|
| 模式一：前缀树的基本骨架 | 208. 实现 Trie（前缀树） | 中等 |
| 模式二：带通配符的搜索 | 211. 添加与搜索单词 - 数据结构设计 | 中等 |
| 模式三：前缀树 + 网格回溯 | 212. 单词搜索 II | 困难 |
| 模式四：找最短词根 | 648. 单词替换 | 中等 |
| 模式五：维护前缀聚合值 | 677. 键值映射 | 中等 |
| 模式六：沿前缀树找最长链 | 720. 词典中最长的单词 | 简单 |
| 模式七：节点挂 Top-K 推荐 | 1268. 搜索推荐系统 | 中等 |
| 模式八：反向插入做后缀匹配 | 1032. 字符流 | 困难 |
| 模式九：双向前缀（前缀 + 后缀） | 745. 前缀和后缀搜索 | 困难 |
| 模式十：0/1 字典树与按位贪心 | 421. 数组中两个数的最大异或值 | 中等 |

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

---

## 模式六：沿前缀树找最长链

**适用信号**：要求在字符串集合里找一个最长的词，且它的**每一个前缀**也都必须是集合里的词。

**核心动作**：把词全部建树后做一次 DFS，只沿着「也是单词」的孩子往下走，走出来的路径天然
满足「每层前缀都是单词」。

### 720. 词典中最长的单词（简单）

**题目**：给定单词数组 `words`，找出其中能由其它单词「每次加一个字母」逐步拼成的最长单词
（即每个前缀都在 `words` 里），并列时取字典序最小者；没有则返回空串。

**思路**：建树时在单词结尾打 `is_end` 标记。从根 DFS：若当前孩子带 `is_end`（说明它是一个
词），才递归下去；每走到一个词节点就按「更长，或等长但字典序更小」更新答案。起点是空串，
所以任何以 `is_end` 开头的第一层字母都能作为链的开端。

**为什么可以整枝砍掉**：题目要求每一步的中间结果都是词。若某个孩子不是词，说明从它再往下
的任何串都会出现「前缀不是词」这一步，全都不合法，于是直接不进入这条分支。这和 212 用
「当前前缀是否在树上」剪枝是同一个思路，只不过这里的剪枝条件更强——必须是完整单词。

**代码**（完整可运行版见 `src/trie/longest_word.py` / `.cpp`）：

```python
def longest_word(words):
    trie = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True

    best = ""

    def dfs(node, path):
        nonlocal best
        if "#" in node:
            if len(path) > len(best) or (len(path) == len(best) and path < best):
                best = path
        for ch, child in node.items():
            if ch != "#" and "#" in child:
                dfs(child, path + ch)

    dfs(trie, "")
    return best
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 26> child;
    bool isEnd = false;
    TrieNode() { child.fill(nullptr); }
};

class LongestWord {
  public:
    std::string longestWord(const std::vector<std::string> &words) {
        TrieNode *root = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root;
            for (char ch : word) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->isEnd = true;
        }
        std::string best;
        dfs(root, "", best);
        return best;
    }

  private:
    void dfs(TrieNode *node, const std::string &path, std::string &best) {
        if (node->isEnd) {
            if (path.size() > best.size() ||
                (path.size() == best.size() && path < best))
                best = path;
        }
        for (int i = 0; i < 26; ++i) {
            TrieNode *child = node->child[i];
            if (child && child->isEnd) dfs(child, path + char('a' + i), best);
        }
    }
};
```

- **复杂度**：建树 O(总字符数)；DFS O(总字符数)。空间 O(总字符数)。
- **易错点**：判严格更长 / 等长更小时要用「长优先、字典序次之」的两段比较，只比长度会
  漏掉并列取字典序的要求；DFS 向下时只走 `is_end` 孩子，但「更新答案」要在进入节点后
  先做（根不是单词，第一次更新发生在第一层）；C++ 递归里 `path + char(...)` 要传新串，
  不要直接改原串。
- **相似题**：212 也是「沿树剪枝」，但那里的条件弱一些（前缀在树上即可）；1268 是同一棵树
  上按前缀取推荐；它体现的「顺序即长度」与 648 找最短词根正好一头一尾。

---

## 模式七：节点挂 Top-K 推荐

**适用信号**：对每个前缀，都要返回「字典序最小的若干个」候选词，且结果随前缀加长而收缩。

**核心动作**：把候选词排序后建树，在每个节点上缓存「最早经过这里的至多 K 个词」，查询时
沿前缀下行、把节点缓存直接抄出来。

### 1268. 搜索推荐系统（中等）

**题目**：给定产品 `products` 与搜索词 `search_word`，用户每多输入一个字母，返回以当前
输入为前缀、字典序最小的至多 3 个产品（不足 3 个就返回全部）。

**思路**：先把 `products` 按字典序排序，再插入前缀树。每到一层就往该节点的推荐列表里追加
当前产品，列表满 3 个就不再追加。查询时逐字符沿树下行，把当前节点的列表复制进答案；一旦
某个字符走不通，后面所有更长前缀都不存在，直接补空列表。

**为什么排个序就够了**：插入顺序就是字典序，某个节点上的 3 个名额会被字典序最小的 3 个词
优先占满；后来的词字典序不可能更小，自然挤不进来。于是每个节点缓存的就是「该前缀下字典序
最小的至多 3 个」，查询无需再排序。这也是「用插入顺序预计算、把查询变廉价」的常见套路。

**代码**（完整可运行版见 `src/trie/suggested_products.py` / `.cpp`）：

```python
def suggested_products(products, search_word):
    trie = {}
    for product in sorted(products):
        node = trie
        for ch in product:
            node = node.setdefault(ch, {})
            node.setdefault("_suggest", [])
            if len(node["_suggest"]) < 3:
                node["_suggest"].append(product)

    result = []
    node = trie
    for ch in search_word:
        if node is not None and ch in node:
            node = node[ch]
            result.append(list(node["_suggest"]))
        else:
            node = None
            result.append([])
    return result
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 26> child;
    std::vector<std::string> suggest;
    TrieNode() { child.fill(nullptr); }
};

class Solution {
  public:
    std::vector<std::vector<std::string>>
    suggestedProducts(std::vector<std::string> products, const std::string &searchWord) {
        std::sort(products.begin(), products.end());
        TrieNode *root = new TrieNode();
        for (const std::string &product : products) {
            TrieNode *node = root;
            for (char ch : product) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
                if (node->suggest.size() < 3) node->suggest.push_back(product);
            }
        }

        std::vector<std::vector<std::string>> result;
        TrieNode *node = root;
        for (char ch : searchWord) {
            if (node) node = node->child[ch - 'a'];
            result.push_back(node ? node->suggest : std::vector<std::string>{});
        }
        return result;
    }
};
```

- **复杂度**：排序 O(n log n)；建树 O(总字符数)；查询 O(L + 答案总量)。空间 O(总字符数)。
- **易错点**：必须**先排序**再插入，否则节点缓存的不一定是字典序最小的 3 个；节点里的
  推荐列表要在每层都追加（不是只在词尾）；查询结果要返回列表的**副本**，避免后续被改动；
  C++ 里 `node->suggest` 是引用，输出时按值拷贝成新 `vector`。
- **相似题**：720 在同一棵树上找最长链；211 是它的通配符版本；347 前 K 个高频元素
  （`heap` 篇）是「在另一维度上取 Top-K」，思路都是「提前维护一小撮候选」。

---

## 模式八：反向插入做后缀匹配

**适用信号**：查询「当前流 / 串的某个**后缀**是否命中给定单词」，且查询是流式的、一次
只追加一个字符。

**核心动作**：把单词**倒着**插入前缀树，于是「后缀匹配」变成「反转串的前缀匹配」，查询
时从最新字符往前沿树走即可。

### 1032. 字符流（困难）

**题目**：初始化给一批单词。每次 `query(letter)` 往流末尾追加一个字母，判断流中是否存在
某个后缀恰好等于给定单词之一。

**思路**：建树时把每个单词逆序插入。查询时先把新字母追加进流，然后从流的最后一个字符开始
往前遍历，同时沿树下行：一旦遇到 `is_end` 节点，说明这一段后缀正好是一个单词，返回真；
中途某个字符不在树上就返回假。

**为什么倒着插**：题目的关键词是「后缀」，而后缀的定义就是「从末尾往前读」。把单词反转后
存进前缀树，流的一个后缀就对应反转流的一个前缀，于是又能用「沿树下行、查 `is_end`」的
标准手法。反过来若正着插、正着走，就得枚举所有起点，代价大得多。

**为什么单次查询不会随流变长**：从末尾往前的路径一旦在树上断开就立即返回，而树的高度就是
最长单词的长度，所以遍历深度有上界，与流的总长无关。流本身只需一直往后追加字符。

**代码**（完整可运行版见 `src/trie/stream_checker.py` / `.cpp`）：

```python
class StreamChecker:
    def __init__(self, words):
        self.trie = {}
        for word in words:
            node = self.trie
            for ch in reversed(word):
                node = node.setdefault(ch, {})
            node["#"] = True
        self.stream = []

    def query(self, letter):
        self.stream.append(letter)
        node = self.trie
        for ch in reversed(self.stream):
            if ch not in node:
                return False
            node = node[ch]
            if node.get("#"):
                return True
        return False
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 26> child;
    bool isEnd = false;
    TrieNode() { child.fill(nullptr); }
};

class StreamChecker {
  public:
    explicit StreamChecker(const std::vector<std::string> &words) {
        root_ = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root_;
            for (auto it = word.rbegin(); it != word.rend(); ++it) {
                int i = *it - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->isEnd = true;
        }
    }

    bool query(char letter) {
        stream_ += letter;
        TrieNode *node = root_;
        for (auto it = stream_.rbegin(); it != stream_.rend(); ++it) {
            int i = *it - 'a';
            if (!node->child[i]) return false;
            node = node->child[i];
            if (node->isEnd) return true;
        }
        return false;
    }

  private:
    TrieNode *root_ = nullptr;
    std::string stream_;
};
```

- **复杂度**：初始化 O(总字符数)；每次查询 O(L)，L 为最长单词长度。空间 O(总字符数)。
- **易错点**：`is_end` 要在每走一步后立刻检查，因为「先出现的更短后缀」也算命中；返回假
  不能清空流，流是持续累积的；单个字母的单词也要能命中（它是自己的后缀）。
- **相似题**：745 也把后缀搬进前缀树，但要多匹配一个前缀；720 用的是正向插入。此题是
  「后缀 → 反向 → 前缀」这条转化的最纯粹例子。

---

## 模式九：双向前缀（前缀 + 后缀）

**适用信号**：同时要求「以某前缀开头」且「以某后缀结尾」，还要在所有命中的词里取某个最优。

**核心动作**：对每个词枚举它的所有后缀 `s`，把 `s + 分隔符 + word` 插入前缀树，插入时沿路
记录下标；查询走 `suffix + 分隔符 + prefix`，一步到位验证两个条件。

### 745. 前缀和后缀搜索（困难）

**题目**：初始化给一批单词。查询 `f(prefix, suffix)` 返回「既是 `prefix` 开头、又是 `suffix`
结尾」的单词中下标最大者的下标，没有返回 -1。

**思路**：分隔符选一个不会出现在单词里的字符（这里用 `'{'`，排在 `'z'` 之后）。对每个词、
每个后缀 `s`，插入 `s + '{' + word`，沿途每个节点都记下当前下标；因为按下标从小到大插入，
节点上留下的就是最大下标。查询时走 `suffix + '{' + prefix`：能走通就返回终点记录的下标，
走不通返回 -1。

**为什么一条路径能同时表达前后缀**：插入的键是 `s + '{' + word`，分隔符把「后缀」和「词身」
分开。查询键里 `'{'` 必须正好落在它与插入键相同的位置上，才能走过分隔符——这就要求插入键的
后缀 `s` 恰是查询的 `suffix`。过了分隔符之后继续走 `prefix`，就是在要求 `word` 以 `prefix`
开头。于是终点节点记录的，正是「后缀、前缀都匹配」的词里下标最大的那个。

**为什么不用两次查询求交集**：分别查前缀、后缀会得到两个集合，再按下标求交既要做集合运算、
又要额外存每个词的索引；把两个条件拼成一条键，一次查询就解决，代价只是初始化时多枚举后缀。

**代码**（完整可运行版见 `src/trie/word_filter.py` / `.cpp`）：

```python
class WordFilter:
    def __init__(self, words):
        self.trie = {}
        for index, word in enumerate(words):
            for start in range(len(word) + 1):
                key = word[start:] + "{" + word
                node = self.trie
                for ch in key:
                    node = node.setdefault(ch, {})
                    node["#"] = index

    def f(self, prefix, suffix):
        node = self.trie
        for ch in suffix + "{" + prefix:
            if ch not in node:
                return -1
            node = node[ch]
        return node.get("#", -1)
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 27> child;  // 0..25 为 a..z，26 存分隔符 '{'
    int best = -1;
    TrieNode() { child.fill(nullptr); }
};

class WordFilter {
  public:
    explicit WordFilter(const std::vector<std::string> &words) {
        root_ = new TrieNode();
        for (int index = 0; index < static_cast<int>(words.size()); ++index) {
            const std::string &word = words[index];
            for (size_t start = 0; start <= word.size(); ++start) {
                std::string key = word.substr(start) + "{" + word;
                TrieNode *node = root_;
                for (char ch : key) {
                    int idx = indexOf(ch);
                    if (!node->child[idx]) node->child[idx] = new TrieNode();
                    node = node->child[idx];
                    node->best = index;
                }
            }
        }
    }

    int f(const std::string &prefix, const std::string &suffix) const {
        TrieNode *node = root_;
        for (char ch : suffix + "{" + prefix) {
            if (!node) return -1;
            node = node->child[indexOf(ch)];
        }
        return node ? node->best : -1;
    }

  private:
    TrieNode *root_ = nullptr;

    static int indexOf(char ch) { return ch == '{' ? 26 : ch - 'a'; }
};
```

- **复杂度**：初始化枚举所有后缀，O(总字符数²)（记为 O(N)）；查询 O(|prefix| + |suffix|)。
  空间 O(总字符数²)。
- **易错点**：分隔符必须选一个不会出现在词里的字符；插入时下标要写在**沿途每个节点**上
  （不是只在词尾），否则查询停在中途会取不到；插入按大小到小顺序可让「后来者覆盖」天然
  得到最大下标；C++ 用 27 长度的数组，分隔符映射到下标 26。
- **相似题**：1032 只匹配后缀，是它的单向版；677 在节点上挂的是聚合值，这里挂的是最优下标，
  都属于「节点额外字段 = 额外能力」。

---

## 模式十：0/1 字典树与按位贪心

**适用信号**：在整数集合里找「异或最大 / 最小」的配对，或者任何按二进制位逐层决策的问题。

**核心动作**：把每个数按「高位到低位」插进一棵每个节点只有 0、1 两个孩子的字典树，求答案
时从高位往低位「尽量走向相反的分支」。

### 421. 数组中两个数的最大异或值（中等）

**题目**：给定整数数组 `nums`，返回 `nums[i] XOR nums[j]` 的最大值。

**思路**：把所有数按二进制从第 31 位到第 0 位插入 0/1 字典树。对每个数 `num`，从高位往
低位走：如果当前位存在与它**相反**的孩子，就走过去，并把答案的这一位记成 1；否则只能走
相同的位，这一位为 0。对所有数取最大值。

**为什么能按位贪心**：异或结果的某一位是 1，当且仅当两个数该位不同。从最高位开始，越高的
位权重越大（是低位的两倍之和还多），所以只要当前位存在相反的分支，选它得到的数一定比不选
更大，低位的选择无关紧要。这就是按位贪心的标准依据。

**为什么比暴力快**：暴力两两异或是 O(n²)。字典树把「和某个数异或最大的搭档」变成一次
「每层尽量走反方向」的 O(位数) 查询，总复杂度降到 O(n·位数)，本质是用空间把逐位决策组织
成一棵树。

**代码**（完整可运行版见 `src/trie/maximum_xor.py` / `.cpp`）：

```python
BITS = 31


def find_maximum_xor(nums):
    trie = {}
    for num in nums:
        node = trie
        for i in range(BITS, -1, -1):
            bit = (num >> i) & 1
            node = node.setdefault(bit, {})

    best = 0
    for num in nums:
        node = trie
        current = 0
        for i in range(BITS, -1, -1):
            bit = (num >> i) & 1
            want = 1 - bit
            if want in node:
                current |= 1 << i
                node = node[want]
            else:
                node = node[bit]
        best = max(best, current)
    return best
```

```cpp
struct TrieNode {
    std::array<TrieNode *, 2> child;
    TrieNode() { child.fill(nullptr); }
};

class Solution {
  public:
    int findMaximumXOR(const std::vector<int> &nums) {
        TrieNode *root = new TrieNode();
        for (int num : nums) {
            TrieNode *node = root;
            for (int i = BITS; i >= 0; --i) {
                int bit = (num >> i) & 1;
                if (!node->child[bit]) node->child[bit] = new TrieNode();
                node = node->child[bit];
            }
        }

        int best = 0;
        for (int num : nums) {
            TrieNode *node = root;
            int current = 0;
            for (int i = BITS; i >= 0; --i) {
                int bit = (num >> i) & 1;
                int want = 1 - bit;
                if (node->child[want]) {
                    current |= 1 << i;
                    node = node->child[want];
                } else {
                    node = node->child[bit];
                }
            }
            if (current > best) best = current;
        }
        return best;
    }

  private:
    static constexpr int BITS = 31;
};
```

- **复杂度**：O(n·B)，B 为位数（本题 31）。空间 O(n·B)。
- **易错点**：位序必须**从高到低**，否则贪心失效；一定要固定位数 `${BITS}`，只遍历到
  最高有效位会让不同数的高度不一致、路径错位；查询时若相反分支不存在，必须退回相同分支
  并继续（不能不更新 `node`）。
- **相似题**：`bit` 篇 136 异或消消乐、338 比特位计数都是按位思考；`trie` 篇 208 把字符
  换成 0/1 就是本题的骨架。凡是对一串「位」做逐层决策，都可以套用 0/1 字典树。

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

10. **后缀问题先想「反转」**：把单词反过来插入，后缀就变成了前缀（1032）。这是一条几乎
    零成本的转化——只要题目问的是「以某段结尾」，就先试试把串反转，把陌生问题归约成
    已经会的前缀问题。

11. **多个条件可以拼进同一条键**：745 把「前缀」和「后缀」用一个不出现在字母表里的
    分隔符拼成一条路径，一次查询同时校验两个条件。遇到「同时满足几个字符串约束」的题，
    不妨想：能不能把它们按固定顺序接成一个键，让「走通这条路径」等价于「全部满足」。
    分隔符的作用是固定边界，防止前后两段互相错位匹配。

12. **查询要反复做，就在建树时预计算**：1268 先把候选排序、在每个节点缓存最小的 3 个，
    查询时直接抄；745 在沿途节点缓存最大下标。把「每次查询都要做的事」提前到插入阶段做
    一次，是前缀树（以及所有索引结构）最常用的提速手段。

13. **字典树的「孩子」不一定是字符**：421 的每个节点只有 0、1 两个孩子，存的是二进制位。
    只要能把对象拆成「一层一层的离散选择」，就能用字典树组织，再配合「从最重要的那一位
    开始贪心」（如按位从高到低）逼近最优解。前缀树是「字符版」，0/1 字典树是「比特版」，
    思想完全一致。
