"""211. 添加与搜索单词 - 数据结构设计

题目：设计一个支持「添加单词」与「搜索单词」的数据结构：
        - add_word(word)：添加单词到数据结构；
        - search(word)：搜索一个单词，word 里可能含通配符 '.'，它能匹配任意单个字母。

思路：仍然是一棵前缀树，插入逻辑与 208 完全一样。不同之处在搜索：普通字符照常匹配，
      遇到 '.' 时，它不对应某一个确定的子节点，而是「当前节点的所有子节点都可以试一试」，
      于是自然写成回溯——对每个子节点递归匹配剩下的部分，任意一条分支成功就返回真。

      为什么 '.' 必须用回溯而不是贪心：因为一个通配符把搜索分叉成多条路径，
      只有走到底才知道哪条能成。当前字符是普通字母时分支唯一，继续往下即可；
      是 '.' 时分叉成「孩子数量」条，需要逐个尝试。这也解释了为什么最坏情况下
      （查询形如 "..." 且树很宽）复杂度会退化，因为本质上是在枚举所有可能路径。

      为什么递归出口在「模式串走完」而不是「节点为空」：判断成功与否看的是
      模式串是否恰好匹配到某个单词结尾，所以当模式串长度走完时，返回当前节点
      的 is_end；而路径不存在时提前返回假。

复杂度：add_word 为 O(L)；search 在含通配符时最坏 O(26^L)，普通查询 O(L)。空间 O(总字符数)。
"""


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


if __name__ == "__main__":
    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")
    assert wd.search("pad") is False
    assert wd.search("bad") is True
    assert wd.search(".ad") is True
    assert wd.search("b..") is True

    wd.add_word("a")
    assert wd.search(".") is True
    assert wd.search("..") is False
    print("add_and_search_words: all tests passed")
