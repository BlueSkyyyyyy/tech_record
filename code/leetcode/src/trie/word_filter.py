"""745. 前缀和后缀搜索

题目：设计一个结构，初始化时给一批单词 words。查询 f(prefix, suffix) 返回
      「既是 prefix 开头、又是 suffix 结尾」的单词中下标最大的那个，没有返回 -1。

思路：对每个单词，枚举它的**每一个后缀** s，把字符串 `s + '{' + word` 插入前缀树
      （`{` 是 'z' 之后的字符，不会出现在单词里，用作分隔符）。插入时在沿途每个
      节点记下当前单词的下标；因为按下标从小到大插入，节点上留下的自然是最大下标。
      查询时走 `suffix + '{' + prefix` 这条路径，走得到就在终点取下标，走不通返回 -1。

      为什么这条路径恰好表达「前、后缀同时匹配」：对某个单词 w、某个后缀 s，
      键是 `s + '{' + w`。查询键 `suffix + '{' + prefix` 想同时经过它们，就必须
      先原样走过 suffix 再遇到分隔符 `{`，这要求 s 就是 suffix（即 w 以 suffix 结尾）；
      随后继续走 prefix，就是在要求 w 以 prefix 开头。于是终点节点汇总的正是
      「后缀、前缀都匹配」的那些单词，取其最大下标即可。

      为什么不用「两次查询求交」：分别查前缀、查后缀会得到两个集合，还要处理
      「带下标求交」，既费空间又费逻辑；把两个条件拼进同一条键，用一次查询解决。

复杂度：初始化枚举所有后缀，O(不同单词的总字符数平方)，记为 O(N)；查询 O(|prefix|+|suffix|)。
"""


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


if __name__ == "__main__":
    wf = WordFilter(["apple"])
    assert wf.f("a", "e") == 0
    assert wf.f("a", "a") == -1
    assert wf.f("b", "") == -1

    wf = WordFilter(["apple", "apply", "banana"])
    assert wf.f("a", "e") == 0
    assert wf.f("a", "y") == 1
    assert wf.f("ban", "na") == 2
    assert wf.f("b", "e") == -1
    # 同前缀同后缀时取下标最大者
    wf = WordFilter(["ab", "b", "ab", "a", "ab"])
    assert wf.f("a", "b") == 4
    print("word_filter: all tests passed")
