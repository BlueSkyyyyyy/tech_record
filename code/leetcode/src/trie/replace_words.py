"""648. 单词替换

题目：给定一个词典 dictionary（若干词根）和一句用空格分隔的句子 sentence，
      把句子里的每个单词替换成它的最短词根：若它可以由某个词根加上若干字母得到，
      就替换成该词根；否则保持原样。

思路：把所有词根插入一棵前缀树。处理句子里的每个单词时，从根出发逐字符往下走，
      一边走一边把走过的字符记成「已匹配前缀」。一旦当前节点是一个词根的结尾，
      说明已经找到「最短的」词根，立刻停下；若中途走不通，就说明这个词没有可用词根。

      为什么「一走到底就停」拿到的就是最短词根：词根在前缀树上是从根往下的路径，
      越先遇到的结尾，路径越短。所以第一个遇到的 isEnd 天然就是最短的那个，
      不需要比较长度。

      为什么不用哈希表逐长度枚举前缀：那样要枚举单词的所有前缀并逐个查表，
      复杂度是 O(L^2)（拼接前缀）或 O(L) 次哈希。前缀树顺着字符一次走完，
      顺便就检查了每个前缀是否是词根，边匹配边判断，更自然。

复杂度：建树 O(总字符数)；处理句子 O(句子总长度)。空间 O(总字符数)。
"""


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


if __name__ == "__main__":
    dictionary = ["cat", "bat", "rat"]
    sentence = "the cattle was rattled by the battery"
    assert replace_words(dictionary, sentence) == "the cat was rat by the bat"

    assert replace_words(["a", "b", "c"], "aadsfasf absbs bbab cadsfafs") == "a a b c"
    assert replace_words(["cat"], "cat") == "cat"
    assert replace_words([], "hello world") == "hello world"
    print("replace_words: all tests passed")
