"""1032. 字符流

题目：设计一个结构，初始化时给一批单词。之后每次 query(letter) 往流里追加一个
      字母，返回「流中某个后缀是否恰好等于某个给定的单词」。

思路：把每个单词**倒着**插入前缀树，这样单词的最后一个字母就成了树里的第一层。
      查询时把新字母追加到流末尾，然后从最新的字母开始、沿「流从后往前」的顺序
      在树里下行：一旦走到某个节点的 is_end，说明流的一段后缀正好是一个单词。

      为什么倒着插、倒着走：题目问的是「后缀」，而后缀的特点是「从末尾往前读」。
      把单词反转后存进前缀树，就把「流的后缀」变成「反转流的某个前缀」，于是又能
      沿用「沿树下行、看 is_end」的老套路。

      为什么每次查询只花 O(最长单词长度)：从末尾往前的路径一旦在树上断开就说明
      没有任何单词能匹配，立刻返回；树的最大深度就是最长单词长度，所以遍历不会
      随流的总长度增长。

复杂度：初始化 O(总字符数)；每次 query O(L)，L 为最长单词长度。空间 O(总字符数)。
"""


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


if __name__ == "__main__":
    checker = StreamChecker(["cd", "f", "kl"])
    letters = "abcdefghijkl"
    expected = [False, False, False, True, False, True, False,
                False, False, False, False, True]
    assert [checker.query(ch) for ch in letters] == expected

    # 单个字母也是合法单词；同一个单词可被后缀重复命中
    checker = StreamChecker(["a"])
    assert checker.query("b") is False
    assert checker.query("a") is True
    assert checker.query("a") is True
    print("stream_checker: all tests passed")
