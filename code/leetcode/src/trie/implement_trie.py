"""208. 实现 Trie（前缀树）

题目：实现一棵前缀树（Trie），支持三个操作：
        - insert(word)：向前缀树中插入字符串 word；
        - search(word)：判断 word 是否已经插入过（必须完整匹配）；
        - starts_with(prefix)：判断是否存在以 prefix 为前缀的已插入字符串。

思路：前缀树的每个节点代表「从根走到这里的这条路径」所对应的字符串前缀。
      节点用一个字典 children 保存「下一个字符 -> 子节点」，再用一个布尔量 is_end
      标记「是否有一个插入过的单词恰好在这里结束」。插入、查找都从根出发，按字符
      逐层往下走：插入时遇到不存在的字符就新建节点，走完后把末节点的 is_end 置真；
      查找时若中途某个字符没有对应子节点就说明不存在。

      为什么用「逐字符分叉 + is_end 标记」而不是直接存到一个集合里：
      集合只能回答「整个单词在不在」，而前缀树的共享前缀结构能顺手回答
      「有没有单词以某前缀开头」，这正是 starts_with 要的。另外，插入的单词越多，
      公共前缀被共享得越多，查询一个长度为 L 的单词只需 O(L) 时间，与单词总数无关。

      为什么 search 和 starts_with 只差最后一步：两者都沿着字符往下走，区别只在
      走完之后——search 要求当前节点 is_end 为真（是一个完整单词），starts_with
      只要节点存在即可（是一条合法前缀）。所以可以把「走路径」抽成一个辅助函数。

复杂度：insert / search / starts_with 均为 O(L)，L 为单词长度；空间 O(总字符数)。
"""


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


if __name__ == "__main__":
    trie = Trie()
    trie.insert("apple")
    assert trie.search("apple") is True
    assert trie.search("app") is False
    assert trie.starts_with("app") is True

    trie.insert("app")
    assert trie.search("app") is True

    trie.insert("banana")
    assert trie.starts_with("ban") is True
    assert trie.search("ban") is False
    assert trie.search("bandana") is False
    print("implement_trie: all tests passed")
